"""
Bioreactor Particle Analyzer
Tests if parameters concentrate or keep particles suspended.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import re


class ParticleAnalyzer:
    def __init__(self, background_path=None, suspended_path=None, background_video_path=None):
        # ── NORMAL HSV RANGES (blue particles) ──────────────────
        # Photo: tighter range, higher quality frames
        self.blue_lower = np.array([100, 50, 50])       # [Hue min, Sat min, Val min]
        self.blue_upper = np.array([125, 255, 255])      # [Hue max, Sat max, Val max]
        # Video: wider range because compression washes out colors
        self.blue_lower_video = np.array([95, 35, 35])   # [Hue min, Sat min, Val min]
        self.blue_upper_video = np.array([130, 255, 255]) # [Hue max, Sat max, Val max]

        # ── CPM HSV RANGES (particles appear more purple/fainter) ──
        # Photo
        self.cpm_lower = np.array([90, 40, 30])          # [Hue min, Sat min, Val min]
        self.cpm_upper = np.array([130, 255, 255])        # [Hue max, Sat max, Val max]
        # Video
        self.cpm_lower_video = np.array([95, 40, 30])    # [Hue min, Sat min, Val min]
        self.cpm_upper_video = np.array([150, 255, 255])  # [Hue max, Sat max, Val max]

        # ── MID-VOLUME HSV RANGES (100mL: slightly wider than normal) ──
        # Photo
        self.midvol_lower = np.array([93, 40, 40])             # [Hue min, Sat min, Val min]
        self.midvol_upper = np.array([135, 255, 255])           # [Hue max, Sat max, Val max]
        # Video
        self.midvol_lower_video = np.array([90, 28, 28])       # [Hue min, Sat min, Val min]
        self.midvol_upper_video = np.array([135, 255, 255])     # [Hue max, Sat max, Val max]

        # ── HIGH-VOLUME HSV RANGES (200mL+: particles fainter through deeper water) ──
        # Photo
        self.highvol_lower = np.array([90, 30, 30])            # [Hue min, Sat min, Val min]
        self.highvol_upper = np.array([140, 255, 255])          # [Hue max, Sat max, Val max]
        # Video
        self.highvol_lower_video = np.array([85, 20, 20])      # [Hue min, Sat min, Val min]
        self.highvol_upper_video = np.array([140, 255, 255])    # [Hue max, Sat max, Val max]

        # ── BACKGROUND SUBTRACTION THRESHOLDS ───────────────────
        # Higher = stricter (less noise, might miss faint particles)
        # Lower = more sensitive (catches faint particles, more noise)
        self.bg_diff_thresh = 15       # Normal mode
        self.highvol_bg_diff_thresh = 15  # High-volume mode (200mL+, blur handles noise)
        self.cpm_bg_diff_thresh = 30  # CPM mode (lower for fainter particles)

        self.background = None
        self.background_video = None
        self.suspended_spread = None
        
        # ── LOAD REFERENCE IMAGES ───────────────────────────────
        if background_path and Path(background_path).exists():
            self.background = cv2.imread(background_path)
            print(f"Loaded background: {background_path}")
        
        if background_video_path and Path(background_video_path).exists():
            cap = cv2.VideoCapture(str(background_video_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.background_video = frame
                print(f"Loaded video background: {background_video_path} ({self.background_video.shape[1]}x{self.background_video.shape[0]})")
        
        if suspended_path and Path(suspended_path).exists():
            suspended = cv2.imread(suspended_path)
            print(f"Loaded suspended reference: {suspended_path}")
            self._suspended_frame = suspended
        else:
            self._suspended_frame = None
        
        # ── INTERNAL STATE ──────────────────────────────────────
        self.mask = None
        self.center = None
        self.radius = None
        self.mask_shape = None
        self.cpm_mode = False          # Toggled per-file based on filename
        self.high_volume = False       # Toggled per-file for 200mL+ (wider HSV)
        self.mid_volume = False        # Toggled per-file for 100mL (slightly wider HSV)
        self._last_cpm_mode = False
        self._last_dilated = None
        self._last_density = None
        self._last_blob_mask = None
        self._detected_circles = {}    # cached per orientation+volume: {'landscape_200': ..., 'portrait_100': ...}

        # Skip early mask init — manual circle draw will trigger on first experiment frame
    
    # ── AUTO-DETECT BIOREACTOR CIRCLE ─────────────────────────
    def _detect_circle(self, frame):
        """Auto-detect the bioreactor circle from a frame.
        Returns (center_x, center_y, radius) or None if detection fails."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = gray.shape[:2]
        min_dim = min(w, h)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate to close small gaps in the vessel edge
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # Find all contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = -1

        for cnt in contours:
            # Fit minimum enclosing circle
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(cx), int(cy), int(r)

            # Filter: radius must be 20-55% of frame min dimension
            if r < min_dim * 0.20 or r > min_dim * 0.55:
                continue

            # Filter: circle must be mostly within frame
            if cx - r < -r * 0.15 or cy - r < -r * 0.15:
                continue
            if cx + r > w + r * 0.15 or cy + r > h + r * 0.15:
                continue

            # Score: prefer large, circular contours
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Score combines size (want largest) and circularity (want roundest)
            score = r * circularity
            if score > best_score:
                best_score = score
                best = (cx, cy, r)

        return best

    # ── MANUAL CIRCLE DRAWING (fallback when auto-detect fails) ──
    def _manual_circle(self, frame):
        """Let user draw a circle by clicking center then dragging to set radius.
        Returns (center_x, center_y, radius) or None if cancelled."""
        display = frame.copy()
        h, w = frame.shape[:2]
        # Scale down for display if too large
        max_dim = 900
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        state = {'center': None, 'radius': 0, 'dragging': False, 'done': False}

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state['center'] = (x, y)
                state['dragging'] = True
                state['radius'] = 0
            elif event == cv2.EVENT_MOUSEMOVE and state['dragging']:
                cx, cy = state['center']
                state['radius'] = int(((x - cx)**2 + (y - cy)**2)**0.5)
            elif event == cv2.EVENT_LBUTTONUP and state['dragging']:
                state['dragging'] = False
                state['done'] = True

        win = 'Draw vessel circle (drag from center outward, ESC to cancel)'
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, mouse_cb)

        print("  Draw the vessel circle: click center, drag to edge, release.")
        print("  Press ESC to skip and use preset fallback.")

        while True:
            vis = display.copy()
            if state['center'] and state['radius'] > 0:
                cv2.circle(vis, state['center'], state['radius'], (0, 255, 0), 2)
                cv2.circle(vis, state['center'], 3, (0, 0, 255), -1)
            cv2.imshow(win, vis)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(win)
                return None
            if state['done']:
                cv2.destroyWindow(win)
                cx, cy = state['center']
                r = state['radius']
                # Scale back to original resolution
                if scale < 1.0:
                    cx = int(cx / scale)
                    cy = int(cy / scale)
                    r = int(r / scale)
                return (cx, cy, r)

    # ── CIRCULAR VESSEL MASK ────────────────────────────────────
    def _init_mask(self, h, w, frame=None, allow_manual=False):
        orientation = 'landscape' if w >= h else 'portrait'
        vol = getattr(self, '_current_volume', 'unknown')
        mode = 'cpm' if self.cpm_mode else vol
        cache_key = f"{orientation}_{mode}"

        # Manual circle drawing only for video frames, per orientation+volume (skip for CPM — uses preset)
        if not self.cpm_mode and frame is not None and cache_key not in self._detected_circles and allow_manual:
            detected = self._manual_circle(frame)
            if detected:
                print(f"  Manual circle ({cache_key}): center=({detected[0]},{detected[1]}), radius={detected[2]}")
                self._detected_circles[cache_key] = (
                    detected[0] / w,    # cx ratio
                    detected[1] / h,    # cy ratio
                    detected[2],        # radius in pixels
                    w, h                # original detection resolution
                )

        if cache_key in self._detected_circles:
            cx_r, cy_r, orig_rad, orig_w, orig_h = self._detected_circles[cache_key]
            self.center = (int(cx_r * w), int(cy_r * h))
            self.radius = int(orig_rad * min(w, h) / min(orig_w, orig_h))
            print(f"  Using drawn vessel ({cache_key}): center=({self.center[0]},{self.center[1]}), radius={self.radius}")
        else:
            # Fallback preset (photos or user cancelled manual draw)
            self.center = (w // 2, h // 2)
            self.radius = int(min(w, h) * 0.47)
            print(f"  Using preset vessel: center=({self.center[0]},{self.center[1]}), radius={self.radius}")

        self.mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(self.mask, self.center, self.radius, 255, -1)
        self.mask_shape = (h, w)
        self._last_cpm_mode = self.cpm_mode
        
        # ── CPM OBSTRUCTION MASK (hub + diagonal bar) ───────────
        if self.cpm_mode:
            obstruction = np.zeros((h, w), dtype=np.uint8)
            # Center offset (tuned at video res 1593x895)
            cpm_center = (self.center[0] + int(w * 6 / 1593),   # ◄ X offset
                          self.center[1] + int(h * 15 / 895))    # ◄ Y offset
            hub_radius = int(self.radius * 0.42)                  # ◄ Hub size (% of vessel radius)
            cv2.circle(obstruction, cpm_center, hub_radius, 255, -1)
            # Diagonal bar
            bar_thickness = int(self.radius * 0.15)               # ◄ Bar width (% of radius)
            angle_rad = np.radians(20)                            # ◄ Bar angle (degrees)
            bar_length = int(self.radius * 2.5)                   # ◄ Bar length (% of radius)
            dx = int(bar_length * np.cos(angle_rad))
            dy = int(bar_length * np.sin(angle_rad))
            pt1 = (cpm_center[0] - dx, cpm_center[1] - dy)
            pt2 = (cpm_center[0] + dx, cpm_center[1] + dy)
            cv2.line(obstruction, pt1, pt2, 255, bar_thickness * 2)
            # Cut obstruction out of vessel mask
            self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(obstruction))
            print(f"  CPM mask applied: hub={hub_radius}px, bar=20deg, {np.sum(obstruction > 0):,}px excluded")
        
    def _needs_mask_reinit(self, h, w):
        """Check if mask needs to be recalculated."""
        if self.mask is None or self.mask_shape != (h, w):
            return True
        if self.cpm_mode != self._last_cpm_mode:
            return True
        return False
    
    # ── PARTICLE DETECTION (core pipeline) ──────────────────────
    def detect_particles(self, frame, video_mode=False):
        h, w = frame.shape[:2]
        if self._needs_mask_reinit(h, w):
            self._init_mask(h, w, frame=frame)
        
        # Step 1: Apply vessel mask
        masked = cv2.bitwise_and(frame, frame, mask=self.mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        
        # Step 2: HSV color thresholding (picks CPM or normal range)
        if video_mode:
            if self.cpm_mode:
                blue_mask = cv2.inRange(hsv, self.cpm_lower_video, self.cpm_upper_video)
            elif self.high_volume:
                blue_mask = cv2.inRange(hsv, self.highvol_lower_video, self.highvol_upper_video)
            elif self.mid_volume:
                blue_mask = cv2.inRange(hsv, self.midvol_lower_video, self.midvol_upper_video)
            else:
                blue_mask = cv2.inRange(hsv, self.blue_lower_video, self.blue_upper_video)
        else:
            if self.cpm_mode:
                blue_mask = cv2.inRange(hsv, self.cpm_lower, self.cpm_upper)
            elif self.high_volume:
                blue_mask = cv2.inRange(hsv, self.highvol_lower, self.highvol_upper)
            elif self.mid_volume:
                blue_mask = cv2.inRange(hsv, self.midvol_lower, self.midvol_upper)
            else:
                blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        # Step 3: Background subtraction (removes static features in vessel)
        if self.background is not None or (video_mode and self.background_video is not None):
            if video_mode and self.background_video is not None:
                bg = self.background_video
            elif self.background is not None:
                bg = self.background
            else:
                bg = None
            if bg is not None:
                if bg.shape[:2] != (h, w):
                    bg = cv2.resize(bg, (w, h))
                bg_masked = cv2.bitwise_and(bg, bg, mask=self.mask)
                diff = cv2.absdiff(masked, bg_masked)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                # Blur to suppress small noise dots while keeping particle regions
                if self.high_volume:
                    diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
                # ◄ Background diff threshold (picks CPM, high-volume, or normal)
                if self.cpm_mode:
                    thresh = self.cpm_bg_diff_thresh
                elif self.high_volume:
                    thresh = self.highvol_bg_diff_thresh
                else:
                    thresh = self.bg_diff_thresh
                _, diff_thresh = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
                blue_mask = cv2.bitwise_and(blue_mask, diff_thresh)

        # Step 4: Re-mask to vessel boundary
        blue_mask = cv2.bitwise_and(blue_mask, self.mask)
        
        # Step 5: Morphological opening for video (removes thin lines/ridges)
        if video_mode:
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # ◄ Opening kernel size
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, open_kernel)
        
        return blue_mask
    
    # ── DEBUG IMAGE SAVER (5-step pipeline visualization) ───────
    def save_debug(self, frame, output_dir, name, video_mode=False):
        """Save each detection step as a separate image for debugging.
        Outputs: {name}_1_masked, _2_hsv_blue, _3_bg_diff, _4_bg_thresh, _5_final"""
        h, w = frame.shape[:2]
        if self._needs_mask_reinit(h, w):
            self._init_mask(h, w, frame=frame)
        
        # Pick HSV range based on mode
        if self.cpm_mode:
            lower = self.cpm_lower_video if video_mode else self.cpm_lower
            upper = self.cpm_upper_video if video_mode else self.cpm_upper
        elif self.high_volume:
            lower = self.highvol_lower_video if video_mode else self.highvol_lower
            upper = self.highvol_upper_video if video_mode else self.highvol_upper
        elif self.mid_volume:
            lower = self.midvol_lower_video if video_mode else self.midvol_lower
            upper = self.midvol_upper_video if video_mode else self.midvol_upper
        else:
            lower = self.blue_lower_video if video_mode else self.blue_lower
            upper = self.blue_upper_video if video_mode else self.blue_upper
        
        # Debug step 1: Masked frame
        masked = cv2.bitwise_and(frame, frame, mask=self.mask)
        cv2.imwrite(str(output_dir / f"{name}_1_masked.png"), masked)
        
        # Debug step 2: HSV blue detection (before background subtraction)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        blue_only = cv2.inRange(hsv, lower, upper)
        hsv_vis = frame.copy()
        hsv_vis[blue_only > 0] = [0, 255, 0]  # Green overlay on detected pixels
        mode_label = "VIDEO" if video_mode else "PHOTO"
        cv2.putText(hsv_vis, f"[{mode_label}] HSV (H:{lower[0]}-{upper[0]} S:{lower[1]}-{upper[1]} V:{lower[2]}-{upper[2]})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(hsv_vis, f"Pixels detected: {np.sum(blue_only > 0):,}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(output_dir / f"{name}_2_hsv_blue.png"), hsv_vis)
        
        # Debug step 3: Background difference
        has_bg = self.background is not None or (video_mode and self.background_video is not None)
        if has_bg:
            if video_mode and self.background_video is not None:
                bg = self.background_video
            else:
                bg = self.background
            if bg.shape[:2] != (h, w):
                bg = cv2.resize(bg, (w, h))
            bg_masked = cv2.bitwise_and(bg, bg, mask=self.mask)
            diff = cv2.absdiff(masked, bg_masked)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Raw difference (amplified for visibility)
            diff_vis = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(str(output_dir / f"{name}_3_bg_diff.png"), diff_vis)
            
            # Debug step 4: Thresholded difference
            if self.cpm_mode:
                thresh = self.cpm_bg_diff_thresh
            elif self.high_volume:
                thresh = self.highvol_bg_diff_thresh
            else:
                thresh = self.bg_diff_thresh
            _, diff_thresh = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
            diff_thresh_vis = frame.copy()
            diff_thresh_vis[diff_thresh > 0] = [0, 0, 255]  # Red overlay on passing pixels
            cv2.putText(diff_thresh_vis, f"BG diff threshold={thresh}, pixels passing: {np.sum(diff_thresh > 0):,}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(output_dir / f"{name}_4_bg_thresh.png"), diff_thresh_vis)
            
            # Final combined (HSV AND bg diff)
            final = cv2.bitwise_and(blue_only, diff_thresh)
            final = cv2.bitwise_and(final, self.mask)
        else:
            final = cv2.bitwise_and(blue_only, self.mask)
        
        # Morphological opening for video
        if video_mode:
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            final = cv2.morphologyEx(final, cv2.MORPH_OPEN, open_kernel)
        
        # Debug step 5: Final detection result
        final_vis = frame.copy()
        final_vis[final > 0] = [0, 255, 0]
        cv2.putText(final_vis, f"Final detection: {np.sum(final > 0):,} pixels",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.circle(final_vis, self.center, self.radius, (0, 255, 255), 2)
        cv2.imwrite(str(output_dir / f"{name}_5_final.png"), final_vis)
        
        print(f"  Debug images saved: {name}_1 through _5")
    
    # ── SPREAD / COVERAGE MEASUREMENT ───────────────────────────
    def calc_spread(self, frame, normalize=True, video_mode=False):
        """Calculate spread using dilated coverage area."""
        blue_mask = self.detect_particles(frame, video_mode=video_mode)
        
        # Dilate to connect nearby particles into coverage regions
        kernel = np.ones((10, 10), np.uint8)  # ◄ Dilation kernel size
        dilated = cv2.dilate(blue_mask, kernel, iterations=2)  # ◄ Dilation iterations
        # Re-mask to prevent dilation from pushing beyond vessel boundary
        dilated = cv2.bitwise_and(dilated, self.mask)
        
        coverage = np.sum(dilated > 0)
        
        # Normalize relative to mask area (same basis as video)
        if normalize:
            mask_area = float(np.sum(self.mask > 0))
            spread = coverage / mask_area if mask_area > 0 else 0.0
        else:
            spread = coverage
        
        self._last_dilated = dilated
        
        return {
            'spread': float(spread),
            'coverage': int(coverage),
            'blue_pixels': int(np.sum(blue_mask > 0))
        }
    
    # ── BLOB ANALYSIS (concentrated particle clusters) ──────────
    def analyze_blobs(self, frame, min_area_ratio=0.005, video_mode=False):
        """Analyze shape and distance of concentrated particle blobs.
        
        All measurements normalized to circular mask:
        - Distances: / mask diameter (0-1)
        - Areas: / mask area (fraction of vessel)
        - Positions: offset from center / radius (-1 to +1)
        """
        spread_result = self.calc_spread(frame, video_mode=video_mode)
        blue_mask = self.detect_particles(frame, video_mode=video_mode)
        
        # Use saturation to separate concentrated (dark teal, high sat) from suspended (light, low sat)
        h, w = frame.shape[:2]
        masked = cv2.bitwise_and(frame, frame, mask=self.mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]  # Saturation channel only
        
        # Only look at saturation where we detected blue particles
        sat_at_particles = np.where(blue_mask > 0, saturation, 0)
        
        # Blur to smooth noise between concentrated blobs
        sat_smooth = cv2.GaussianBlur(sat_at_particles.astype(np.float32),
                                       (101, 101), 0)  # ◄ Blur kernel (bigger = smoother blobs)
        
        # Normalize to 0-255
        if sat_smooth.max() > 0:
            sat_norm = (sat_smooth / sat_smooth.max() * 255).astype(np.uint8)
        else:
            sat_norm = np.zeros_like(blue_mask)
        
        # Threshold to keep only intense concentrated cores
        _, conc_mask = cv2.threshold(sat_norm, 120, 255, cv2.THRESH_BINARY)  # ◄ Blob saturation threshold
        
        # Count pixels in concentrated vs suspended regions
        total_blue = int(np.sum(blue_mask > 0))
        concentrated_pixels = int(np.sum((blue_mask > 0) & (conc_mask > 0)))
        suspended_pixels = total_blue - concentrated_pixels
        
        # Store for debug/overlay
        self._last_density = sat_norm
        self._last_blob_mask = conc_mask
        
        # ── CONTOUR ANALYSIS on concentrated regions ────────────
        dilated = conc_mask
        diameter = self.radius * 2
        mask_area = np.pi * self.radius ** 2
        cx, cy = self.center
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = min_area_ratio * mask_area  # ◄ Min blob size (fraction of vessel area)
        blobs = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            bx = M['m10'] / M['m00']
            by = M['m01'] / M['m00']
            
            rel_x = (bx - cx) / self.radius       # Position: -1 (left) to +1 (right)
            rel_y = (by - cy) / self.radius        # Position: -1 (top) to +1 (bottom)
            dist_from_center = np.sqrt((bx - cx)**2 + (by - cy)**2) / self.radius  # 0=center, 1=edge
            
            area_ratio = area / mask_area          # Blob area as fraction of vessel
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0  # 1.0 = perfect circle
            
            (_, _), enc_radius = cv2.minEnclosingCircle(cnt)
            enc_radius_ratio = enc_radius / self.radius
            
            blob = {
                'centroid': (float(bx), float(by)),
                'rel_position': (round(float(rel_x), 3), round(float(rel_y), 3)),
                'dist_from_center': round(float(dist_from_center), 3),
                'area_ratio': round(float(area_ratio), 4),
                'circularity': round(float(circularity), 3),
                'enc_radius_ratio': round(float(enc_radius_ratio), 3),
            }
            
            # Ellipse fit (orientation info)
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (_, (maj, mi), angle) = ellipse
                blob['major_axis_ratio'] = round(float(max(maj, mi)) / diameter, 3)
                blob['minor_axis_ratio'] = round(float(min(maj, mi)) / diameter, 3)
                blob['orientation_deg'] = round(float(angle), 1)
                blob['elongation'] = round(float(max(maj, mi)) / max(min(maj, mi), 1), 2)
            
            blobs.append(blob)
        
        blobs.sort(key=lambda b: b['area_ratio'], reverse=True)
        
        # Inter-blob distances (normalized to diameter)
        inter_distances = []
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                c1 = blobs[i]['centroid']
                c2 = blobs[j]['centroid']
                dist = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2) / diameter
                inter_distances.append({
                    'blob_pair': (i, j),
                    'distance': round(float(dist), 3)
                })
        
        # ── PATTERN CLASSIFICATION ──────────────────────────────
        n = len(blobs)
        if n == 0:
            pattern = 'NO_PARTICLES'
        elif n == 1:
            if blobs[0]['dist_from_center'] < 0.3:   # ◄ Center vs offset threshold
                pattern = 'SINGLE_CENTER'
            else:
                pattern = 'SINGLE_OFFSET'
        elif n == 2:
            max_dist = max(d['distance'] for d in inter_distances) if inter_distances else 0
            if max_dist > 0.6:                        # ◄ Split vs near threshold
                pattern = 'TWO_BLOB_SPLIT'
            else:
                pattern = 'TWO_BLOB_NEAR'
        else:
            pattern = 'MULTI_CLUSTER'
        
        return {
            'spread': spread_result['spread'],
            'num_blobs': n,
            'pattern': pattern,
            'blobs': blobs,
            'inter_distances': inter_distances,
            'total_blob_area_ratio': round(sum(b['area_ratio'] for b in blobs), 4),
            'total_blue_pixels': total_blue,
            'concentrated_pixels': concentrated_pixels,
            'suspended_pixels': suspended_pixels,
            'concentration_ratio': round(concentrated_pixels / max(total_blue, 1), 3),
        }
    
    # ── VIDEO ANALYSIS (end-state coverage) ─────────────────────
    def analyze_video(self, path):
        """Analyze video end state as coverage % of mask area."""
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total == 0:
            cap.release()
            return None
        
        # Read first frame to init mask
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        h, w = frame.shape[:2]
        if self._needs_mask_reinit(h, w):
            self._init_mask(h, w, frame=frame, allow_manual=True)
        mask_area = float(np.sum(self.mask > 0))
        
        # Average last 3 frames for stability
        end_coverages = []
        last_frame = None
        for i in range(max(0, total-3), total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                end_coverages.append(self.calc_spread(frame, normalize=False, video_mode=True)['coverage'])
                last_frame = frame
        
        cap.release()
        
        if not end_coverages:
            return None
        
        end_pct = np.mean(end_coverages) / mask_area
        
        return {
            'end_pct': float(end_pct),
            'mask_area': int(mask_area),
            'total_frames': total,
            'last_frame': last_frame
        }
    
    # ── PHOTO vs VIDEO FIRST FRAME COMPARISON ───────────────────
    def compare_photo_to_video_start(self, photo_path, video_path):
        """Compare end photo to video first frame. Returns change relative to first frame = 100%.
        WARNING: photo-vs-video comparison is unreliable due to resolution mismatch."""
        saved_mask = self.mask
        saved_center = self.center
        saved_radius = self.radius
        saved_shape = self.mask_shape
        saved_suspended = self.suspended_spread
        
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            return None

        vid_h, vid_w = first_frame.shape[:2]

        photo = cv2.imread(str(photo_path))
        if photo is None:
            return None
        
        # Resize photo to video res + compress to match video quality
        photo_resized = cv2.resize(photo, (vid_w, vid_h))
        _, encoded = cv2.imencode('.jpg', photo_resized, [cv2.IMWRITE_JPEG_QUALITY, 50])  # ◄ JPEG quality
        photo_resized = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        self._init_mask(vid_h, vid_w)
        mask_area = np.sum(self.mask > 0)
        
        start_coverage = self.calc_spread(first_frame, normalize=False, video_mode=True)['coverage'] / mask_area
        photo_coverage = self.calc_spread(photo_resized, normalize=False, video_mode=True)['coverage'] / mask_area
        
        # Restore state
        self.mask = saved_mask
        self.center = saved_center
        self.radius = saved_radius
        self.mask_shape = saved_shape
        self.suspended_spread = saved_suspended
        
        if start_coverage == 0:
            return None
        
        change = min((photo_coverage / start_coverage) - 1.0, 0.0)
        
        return {
            'photo_change': float(change),
            'start_pct': float(start_coverage),
            'photo_pct': float(photo_coverage),
        }
    
    # ── SUSPENDED CONTENT MEASUREMENT ───────────────────────────
    def measure_suspended_content(self, frame, conc_mask=None):
        """Measure fraction of particle content NOT in concentrated blobs.
        Uses saturation intensity (no resolution mismatch).
        Returns ratio: 1.0 = everything suspended, 0.0 = everything in blobs."""
        h, w = frame.shape[:2]
        if self._needs_mask_reinit(h, w):
            self._init_mask(h, w, frame=frame)
        
        # Get background for subtraction
        bg = None
        if self.background is not None:
            bg = self.background
            if bg.shape[:2] != (h, w):
                bg = cv2.resize(bg, (w, h))
        
        # Background-subtracted saturation
        masked = cv2.bitwise_and(frame, frame, mask=self.mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        
        if bg is not None:
            bg_masked = cv2.bitwise_and(bg, bg, mask=self.mask)
            bg_hsv = cv2.cvtColor(bg_masked, cv2.COLOR_BGR2HSV)
            bg_sat = bg_hsv[:, :, 1].astype(np.float32)
            sat = np.maximum(sat - bg_sat, 0)
        
        sat = np.where(self.mask > 0, sat, 0)
        total_sat = float(np.sum(sat))
        
        if total_sat == 0:
            return None
        
        # Split into concentrated vs non-concentrated zones
        if conc_mask is not None:
            if conc_mask.shape[:2] != (h, w):
                conc_mask = cv2.resize(conc_mask, (w, h))
            non_conc_mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(conc_mask))
        else:
            non_conc_mask = self.mask
        
        non_conc_sat = float(np.sum(np.where(non_conc_mask > 0, sat, 0)))
        conc_sat = total_sat - non_conc_sat
        ratio = non_conc_sat / total_sat
        
        result = {
            'suspended_ratio': round(float(ratio), 3),
            'concentrated_ratio': round(float(1 - ratio), 3),
            'total_saturation': round(total_sat, 0),
            'non_conc_saturation': round(non_conc_sat, 0),
            'conc_saturation': round(conc_sat, 0),
        }
        
        # Compare against Suspended reference photo
        if self._suspended_frame is not None:
            susp_frame = self._suspended_frame
            if susp_frame.shape[:2] != (h, w):
                susp_frame = cv2.resize(susp_frame, (w, h))
            susp_masked = cv2.bitwise_and(susp_frame, susp_frame, mask=self.mask)
            susp_hsv = cv2.cvtColor(susp_masked, cv2.COLOR_BGR2HSV)
            susp_sat = susp_hsv[:, :, 1].astype(np.float32)
            if bg is not None:
                susp_sat = np.maximum(susp_sat - bg_sat, 0)
            susp_sat = np.where(self.mask > 0, susp_sat, 0)
            susp_total = float(np.sum(susp_sat))
            mask_pixels = float(np.sum(self.mask > 0))
            non_conc_pixels = float(np.sum(non_conc_mask > 0))
            
            if susp_total > 0 and mask_pixels > 0 and non_conc_pixels > 0:
                result['vs_suspended_total'] = round(min(float(non_conc_sat / susp_total), 1.0), 3)
                result['suspended_ref_total'] = round(susp_total, 0)
        
        return result
    
    # ── OVERLAY IMAGE SAVER ─────────────────────────────────────
    def save_overlay(self, frame, path, draw_blobs=False, video_mode=False):
        blue_mask = self.detect_particles(frame, video_mode=video_mode)
        overlay = frame.copy()
        overlay[blue_mask > 0] = [0, 255, 0]  # Green = detected particles
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.circle(result, self.center, self.radius, (0, 255, 255), 2)  # Yellow vessel outline
        
        spread = self.calc_spread(frame, video_mode=video_mode)
        text = f"Spread: {spread['spread']:.0%}"
        cv2.putText(result, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(result, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # Blob contour overlay (red outlines + centroids)
        if draw_blobs:
            blob_result = self.analyze_blobs(frame, video_mode=video_mode)
            blue_mask_blobs = self.detect_particles(frame, video_mode=video_mode)
            masked_blobs = cv2.bitwise_and(frame, frame, mask=self.mask)
            hsv_blobs = cv2.cvtColor(masked_blobs, cv2.COLOR_BGR2HSV)
            sat = hsv_blobs[:, :, 1]
            sat_at_particles = np.where(blue_mask_blobs > 0, sat, 0)
            sat_smooth = cv2.GaussianBlur(sat_at_particles.astype(np.float32), (101, 101), 0)
            if sat_smooth.max() > 0:
                sat_norm = (sat_smooth / sat_smooth.max() * 255).astype(np.uint8)
            else:
                sat_norm = np.zeros_like(blue_mask_blobs)
            _, blob_dilated = cv2.threshold(sat_norm, 120, 255, cv2.THRESH_BINARY)  # ◄ Must match blob threshold
            contours, _ = cv2.findContours(blob_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 0.005 * np.pi * self.radius ** 2  # ◄ Must match min_area_ratio
            
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue
                cv2.drawContours(result, [cnt], -1, (0, 0, 255), 2)  # Red blob outlines
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    bx = int(M['m10'] / M['m00'])
                    by = int(M['m01'] / M['m00'])
                    cv2.circle(result, (bx, by), 6, (0, 0, 255), -1)
                    cv2.circle(result, (bx, by), 6, (255, 255, 255), 2)
                    cv2.line(result, self.center, (bx, by), (0, 0, 255), 1, cv2.LINE_AA)
            
            pattern_text = f"Pattern: {blob_result['pattern']}"
            cv2.putText(result, pattern_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(result, pattern_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imwrite(str(path), result)


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    # ==================== SETTINGS ====================
    SAVE_IMAGES = False  # Save photo overlays + comparison chart
    SAVE_VIDEO = True    # Save video frame detection overlays
    TEST_SINGLE = False  # Only process one image (for debugging)
    TEST_NAME = ""  # Filename stem (empty = first)
    TEST_CPM = False  # Only process CPM files
    # ==================================================
    
    parser = argparse.ArgumentParser(description='Bioreactor Particle Analyzer')
    parser.add_argument('--input', '-i', default='Mixing time', help='Input folder')
    parser.add_argument('--output', '-o', default='./output', help='Output folder')
    parser.add_argument('--cpm', action='store_true', help='Only process CPM files')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.cpm:  # CLI --cpm overrides manual setting
        TEST_CPM = True
    
    if not input_dir.exists():
        print(f"Folder not found: {input_dir}")
        return
    
    # ── FIND REFERENCE FILES ────────────────────────────────────
    # Background photo (for photo bg subtraction)
    bg_path = None
    for name in ['Background.JPG', 'Background.jpg', 'background.JPG']:
        if (input_dir / name).exists():
            bg_path = str(input_dir / name)
            break
    
    # Background video - normal (for video bg subtraction)
    bg_video_path = None
    for name in ['Background video.MP4', 'Background video.mp4', 'Background_video.MP4', 'Background_video.mp4',
                  'Background video no water.MP4', 'Background video no water.mp4']:
        if (input_dir / name).exists():
            bg_video_path = str(input_dir / name)
            break
    
    # Background video - CPM (base_before_adding_particles.mp4)
    bg_video_cpm_path = None
    for name in ['base_before_adding_particles.mp4', 'base_before_adding_particles.MP4']:
        if (input_dir / name).exists():
            bg_video_cpm_path = str(input_dir / name)
            break
    
    # Suspended reference photo (= 100% dispersed baseline)
    susp_path = None
    for name in ['Suspended.JPG', 'Suspended.jpg', 'suspended.JPG']:
        if (input_dir / name).exists():
            susp_path = str(input_dir / name)
            break
    
    analyzer = ParticleAnalyzer(bg_path, susp_path, bg_video_path)
    
    # Load CPM background video separately (swapped in when processing CPM files)
    bg_video_cpm = None
    if bg_video_cpm_path:
        cap = cv2.VideoCapture(bg_video_cpm_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            bg_video_cpm = frame
            print(f"Loaded CPM video background: {bg_video_cpm_path} ({frame.shape[1]}x{frame.shape[0]})")
    
    # Store normal background so we can swap back from CPM
    analyzer._bg_video_normal = analyzer.background_video

    # ── VOLUME-SPECIFIC BLANK REFERENCES ──────────────────────
    volume_backgrounds = {}  # {volume_str: frame}
    ref_paths = set(input_dir.glob('*Blank Reference*')) | set(input_dir.glob('*blank reference*'))
    for ref_path in ref_paths:
        m = re.search(r'(\d+)\s*m[lL]', ref_path.stem)
        if m:
            vol = m.group(1)  # e.g. "200"
            cap = cv2.VideoCapture(str(ref_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                volume_backgrounds[vol] = frame
                print(f"Loaded volume background: {ref_path.name} → {vol}mL ({frame.shape[1]}x{frame.shape[0]})")

    # ── FIND EXPERIMENT FILES ───────────────────────────────────
    exclude = ['background', 'suspended', 'empty', 'base_before', 'blank', 'reference', '_debug']  # ◄ Filenames to skip
    images = []
    for ext in ['*.JPG', '*.jpg', '*.png', '*.PNG']:
        images += [p for p in input_dir.glob(ext) if not any(e in p.stem.lower() for e in exclude)]
    images = sorted(set(images))
    
    # Filter: CPM only
    if TEST_CPM:
        images = [p for p in images if 'cpm' in p.stem.lower()]
        print(f"CPM mode: filtering to CPM files only")
    
    # Filter: single test
    if TEST_SINGLE:
        if TEST_NAME:
            images = [p for p in images if p.stem == TEST_NAME]
        else:
            images = images[:1]
    
    videos = {p.stem: p for p in input_dir.glob('*.mp4') if not any(e in p.stem.lower() for e in exclude)}
    videos.update({p.stem: p for p in input_dir.glob('*.MP4') if not any(e in p.stem.lower() for e in exclude)})
    
    if TEST_CPM:  # Filter videos too
        videos = {k: v for k, v in videos.items() if 'cpm' in k.lower()}
    
    print(f"Found {len(images)} images, {len(videos)} videos\n")
    
    # ════════════════════════════════════════════════════════════
    #  MAIN PROCESSING LOOP
    # ════════════════════════════════════════════════════════════
    results = []
    
    for img_path in images:
        name = img_path.stem
        print(f"\n{name}")
        print("-" * 40)
        
        # ── CPM MODE TOGGLE (auto based on filename) ───────────
        is_cpm = 'cpm' in name.lower()
        if is_cpm != analyzer.cpm_mode:
            analyzer.cpm_mode = is_cpm
            analyzer.mask = None  # Force mask reinit
            if is_cpm:
                if bg_video_cpm is not None:
                    analyzer.background_video = bg_video_cpm  # ◄ Swap to CPM background
                print(f"  CPM mode enabled (CPM background + purple HSV range)")
            else:
                analyzer.background_video = analyzer._bg_video_normal  # ◄ Swap back to normal

        # ── VOLUME-BASED BACKGROUND SWAP + HSV RANGE ────────
        vol_match = re.search(r'(\d+)\s*m[lL]', name)
        if vol_match:
            vol_num = int(vol_match.group(1))
            analyzer.high_volume = vol_num >= 200
            analyzer.mid_volume = 100 <= vol_num < 200
            new_vol = vol_match.group(1)
            if getattr(analyzer, '_current_volume', None) != new_vol:
                analyzer.mask = None  # Force mask reinit for new volume
            analyzer._current_volume = new_vol
            if volume_backgrounds and not is_cpm and new_vol in volume_backgrounds:
                analyzer.background_video = volume_backgrounds[new_vol]
        else:
            analyzer.high_volume = False
            analyzer.mid_volume = False
            analyzer._current_volume = 'unknown'

        # ── ANALYZE END PHOTO ──────────────────────────────────
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Could not read image")
            continue

        img_spread = analyzer.calc_spread(frame)
        
        result = {
            'name': name,
            'end_spread': img_spread['spread'],
            'blue_pixels': img_spread['blue_pixels']
        }
        
        # ── ANALYZE MATCHING VIDEO ─────────────────────────────
        first_frame = None
        last_video_frame = None
        if name in videos:
            vid_result = analyzer.analyze_video(videos[name])
            if vid_result:
                result['video_end_pct'] = vid_result['end_pct']
                last_video_frame = vid_result.get('last_frame')
                
                # Read first frame for debug
                cap = cv2.VideoCapture(str(videos[name]))
                ret, first_frame = cap.read()
                cap.release()
                if not ret:
                    first_frame = None
                
                # ── SAVE VIDEO FRAME OVERLAYS ──────────────────
                if SAVE_VIDEO:
                    # Save video overlays later (after classification) so we know whether to draw blobs
                    _save_video_frames = (first_frame, last_video_frame)
                    if TEST_SINGLE:  # Full debug steps for single test
                        if first_frame is not None:
                            analyzer.save_debug(first_frame, output_dir, f"{name}_vid_first", video_mode=True)
                            print(f"  Saved first frame debug")
                        if last_video_frame is not None:
                            analyzer.save_debug(last_video_frame, output_dir, f"{name}_vid_last", video_mode=True)
                            print(f"  Saved last frame debug")
                
                print(f"  Video end: {vid_result['end_pct']:.1%} of mask")
            else:
                print(f"  Video: could not read")
        
        # ── PHOTO END COVERAGE ─────────────────────────────────
        h, w = frame.shape[:2]
        if analyzer._needs_mask_reinit(h, w):
            analyzer._init_mask(h, w)
        photo_mask_area = float(np.sum(analyzer.mask > 0))
        
        end_coverage = analyzer.calc_spread(frame, normalize=False)['coverage']
        photo_end_pct = end_coverage / photo_mask_area
        result['photo_end_pct'] = photo_end_pct
        print(f"  Photo end: {photo_end_pct:.1%} of mask")
        
        # ── CLASSIFICATION (based on video end coverage) ───────
        end_pct = result.get('video_end_pct')
        if end_pct is not None:
            if is_cpm:
                # CPM: detection only catches concentrated particles
                # so lower thresholds — high detection = concentrated
                if end_pct < 0.05:       # ◄ CPM: < 5% = suspended (barely visible)
                    result['status'] = 'MOSTLY_SUSPENDED'
                elif end_pct < 0.135:  # ◄ CPM: 5-13.5% = intermediate
                    result['status'] = 'INTERMEDIATE'
                else:                     # ◄ CPM: > 13.5% = concentrating
                    result['status'] = 'CONCENTRATING'
            elif analyzer.high_volume:
                if end_pct < 0.35:       # ◄ 200mL+ CONCENTRATING cutoff
                    result['status'] = 'CONCENTRATING'
                elif end_pct < 0.55:     # ◄ 200mL+ INTERMEDIATE cutoff
                    result['status'] = 'INTERMEDIATE'
                else:                     # ◄ 200mL+ MOSTLY_SUSPENDED
                    result['status'] = 'MOSTLY_SUSPENDED'
            elif analyzer.mid_volume:
                if end_pct < 0.35:       # ◄ 100mL CONCENTRATING cutoff
                    result['status'] = 'CONCENTRATING'
                elif end_pct < 0.46:     # ◄ 100mL INTERMEDIATE cutoff
                    result['status'] = 'INTERMEDIATE'
                else:                     # ◄ 100mL MOSTLY_SUSPENDED
                    result['status'] = 'MOSTLY_SUSPENDED'
            else:
                if end_pct < 0.40:       # ◄ CONCENTRATING cutoff
                    result['status'] = 'CONCENTRATING'
                elif end_pct < 0.80:     # ◄ INTERMEDIATE cutoff
                    result['status'] = 'INTERMEDIATE'
                else:                     # ◄ MOSTLY_SUSPENDED (everything above)
                    result['status'] = 'MOSTLY_SUSPENDED'
            print(f"  Status: {result['status']}")
        
        # ── BLOB ANALYSIS (on photo, for non-concentrating; skip CPM) ────
        if not analyzer.cpm_mode and (result.get('status') in ('INTERMEDIATE', 'MOSTLY_SUSPENDED') or TEST_SINGLE):
            blob_result = analyzer.analyze_blobs(frame)
            result['blob_analysis'] = {
                'num_blobs': blob_result['num_blobs'],
                'pattern': blob_result['pattern'],
                'total_blob_area_ratio': blob_result['total_blob_area_ratio'],
                'blobs': blob_result['blobs'],
                'inter_distances': blob_result['inter_distances'],
                'concentration_ratio': blob_result['concentration_ratio'],
                'concentrated_pixels': blob_result['concentrated_pixels'],
                'suspended_pixels': blob_result['suspended_pixels'],
            }
            print(f"  Pattern: {blob_result['pattern']} ({blob_result['num_blobs']} blobs)")
            print(f"  Concentration ratio: {blob_result['concentration_ratio']:.0%} concentrated / {1 - blob_result['concentration_ratio']:.0%} suspended")
            print(f"  Total blob area: {blob_result['total_blob_area_ratio']:.1%} of vessel")
            
            # ── SUSPENDED CONTENT MEASUREMENT ──────────────────
            if hasattr(analyzer, '_last_blob_mask'):
                susp_content = analyzer.measure_suspended_content(
                    frame, analyzer._last_blob_mask)
                if susp_content:
                    result['suspended_content'] = susp_content
                    print(f"  Suspended content: {susp_content['suspended_ratio']:.0%} suspended / {susp_content['concentrated_ratio']:.0%} concentrated")
                    print(f"    (Non-conc sat: {susp_content['non_conc_saturation']:,.0f} / Total: {susp_content['total_saturation']:,.0f})")
                    if 'vs_suspended_total' in susp_content:
                        print(f"  vs Suspended ref: {susp_content['vs_suspended_total']:.0%} of fully suspended")
            
            # Print per-blob details
            for i, b in enumerate(blob_result['blobs']):
                print(f"    Blob {i}: pos=({b['rel_position'][0]:+.2f}, {b['rel_position'][1]:+.2f})"
                      f"  area={b['area_ratio']:.1%}  circ={b['circularity']:.2f}"
                      f"  dist_from_center={b['dist_from_center']:.2f}")
                if 'elongation' in b:
                    print(f"           elong={b['elongation']:.1f}x  angle={b['orientation_deg']:.0f}°")
            for d in blob_result['inter_distances']:
                print(f"    Blob {d['blob_pair'][0]}<->{d['blob_pair'][1]}: {d['distance']:.2f} diameters apart")
            
            # Save density debug images
            if TEST_SINGLE and SAVE_IMAGES and hasattr(analyzer, '_last_density'):
                sat_color = cv2.applyColorMap(analyzer._last_density, cv2.COLORMAP_JET)
                cv2.imwrite(str(output_dir / f"{name}_saturation_map.png"), sat_color)
                if hasattr(analyzer, '_last_blob_mask'):
                    cv2.imwrite(str(output_dir / f"{name}_blob_mask.png"), analyzer._last_blob_mask)
                print(f"  Saved saturation map + blob mask")
        
        # ── SAVE PHOTO OVERLAY ─────────────────────────────────
        if SAVE_IMAGES:
            is_not_concentrating = result.get('status') in ('INTERMEDIATE', 'MOSTLY_SUSPENDED')
            analyzer.save_overlay(frame, output_dir / f"{name}_detection.png", draw_blobs=(not analyzer.cpm_mode and (is_not_concentrating or TEST_SINGLE)))
            if TEST_SINGLE:
                analyzer.save_debug(frame, output_dir, f"{name}_photo")

        # ── SAVE VIDEO OVERLAYS (after classification) ─────
        if SAVE_VIDEO and '_save_video_frames' in dir():
            first_vf, last_vf = _save_video_frames
            show_blobs = not analyzer.cpm_mode and (result.get('status') != 'MOSTLY_SUSPENDED' or TEST_SINGLE)
            if last_vf is not None:
                cv2.imwrite(str(output_dir / f"{name}_video_raw.png"), last_vf)
                analyzer.save_overlay(last_vf, output_dir / f"{name}_video_detection.png", draw_blobs=show_blobs, video_mode=True)
                print(f"  Saved video last frame (raw + overlay)")
            if first_vf is not None:
                analyzer.save_overlay(first_vf, output_dir / f"{name}_first_frame.png", draw_blobs=False, video_mode=True)
                print(f"  Saved video first frame overlay")
            del _save_video_frames

        results.append(result)
    
    if not results:
        print("No images analyzed!")
        return
    
    # ── SAVE RESULTS JSON ──────────────────────────────────────
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ── COMPARISON CHART ───────────────────────────────────────
    if SAVE_IMAGES:
        names = [r['name'][:15] for r in results]
        has_video = any('video_end_pct' in r for r in results)
        fig_width = max(12, len(results) * 0.8)
        
        if has_video:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 12))
            
            # Video end coverage chart
            video_results = [r for r in results if 'video_end_pct' in r]
            v_names = [r['name'][:15] for r in video_results]
            v_pcts = [r['video_end_pct'] for r in video_results]
            
            # ◄ Chart color thresholds (NOTE: out of sync with classification 50%/80%)
            colors = ['red' if p < 0.40 else 'orange' if p < 0.80 else 'green' for p in v_pcts]
            ax1.bar(range(len(v_names)), v_pcts, color=colors, edgecolor='black')
            ax1.axhline(0.40, color='red', linestyle='--', alpha=0.5, label='Concentrating < 40%')    # ◄ Chart dashed line
            ax1.axhline(0.80, color='orange', linestyle='--', alpha=0.5, label='Intermediate < 80%')  # ◄ Chart dashed line
            ax1.set_ylabel('Coverage (% of mask)')
            ax1.set_title('Video End: Coverage as % of Mask Area')
            ax1.set_xticks(range(len(v_names)))
            ax1.set_xticklabels(v_names, rotation=45, ha='right', fontsize=8)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax1.legend(fontsize=8)
            
            # Photo end coverage chart
            photo_pcts = [r.get('photo_end_pct', 0) for r in results]
            colors2 = ['red' if p < 0.40 else 'orange' if p < 0.80 else 'green' for p in photo_pcts]
            ax2.bar(range(len(names)), photo_pcts, color=colors2, edgecolor='black')
            ax2.axhline(0.40, color='red', linestyle='--', alpha=0.5, label='Concentrating < 40%')
            ax2.axhline(0.80, color='orange', linestyle='--', alpha=0.5, label='Intermediate < 80%')
            ax2.set_ylabel('Coverage (% of mask)')
            ax2.set_title('End Photo: Coverage as % of Mask Area')
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax2.legend(fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            photo_pcts = [r.get('photo_end_pct', 0) for r in results]
            colors2 = ['red' if p < 0.40 else 'orange' if p < 0.80 else 'green' for p in photo_pcts]
            ax.bar(range(len(names)), photo_pcts, color=colors2, edgecolor='black')
            ax.axhline(0.40, color='red', linestyle='--', alpha=0.5, label='Concentrating < 40%')
            ax.axhline(0.80, color='orange', linestyle='--', alpha=0.5, label='Intermediate < 80%')
            ax.set_ylabel('Coverage (% of mask)')
            ax.set_title('End Photo: Coverage as % of Mask Area')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison.png', dpi=150)
    
    # ── SUMMARY TABLE ──────────────────────────────────────────
    print(f"\n{'='*85}")
    print("SUMMARY")
    print('='*85)
    print(f"{'Name':<15} {'VidEnd':>8} {'PhotoEnd':>8} {'Conc%':>8} {'SuspCont':>8} {'Status':<15}")
    print('-'*85)
    for r in results:
        vid = f"{r['video_end_pct']:.1%}" if 'video_end_pct' in r else "N/A"
        photo = f"{r['photo_end_pct']:.1%}" if 'photo_end_pct' in r else "N/A"
        conc = f"{r['blob_analysis']['concentration_ratio']:.0%}" if 'blob_analysis' in r else "N/A"
        susp = f"{r['suspended_content']['suspended_ratio']:.0%}" if 'suspended_content' in r else "N/A"
        status = r.get('status', '')
        print(f"{r['name'][:15]:<15} {vid:>8} {photo:>8} {conc:>8} {susp:>8} {status:<15}")
    print('='*85)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()