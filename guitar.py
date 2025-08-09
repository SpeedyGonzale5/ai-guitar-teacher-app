import cv2
import mediapipe as mp
import numpy as np
import json
import re
import os
import time
from typing import Dict, List, Optional, Tuple

mp_drawing = mp.solutions.drawing_utils


# Configuration for different video/JSON combinations
CONFIGS: Dict[int, Dict[str, str]] = {
    1: {
        'json_file': 'guitar.json',
        'video_file': 'ai_guitar_video.mp4',
        'output_file': 'guitar_feedback.mp4',
    }
}

# Choose which configuration to use (change this number: 1..N)
CURRENT_CONFIG: int = 1
config = CONFIGS[CURRENT_CONFIG]


def parse_timestamp(timestamp: str) -> float:
    """Parse flexible timestamps like 'm:ss', 'm:ss.s', '1m23s', '83s'."""
    t = (timestamp or '0:00').strip().lower()
    # Handle patterns like '1m23s' or '83s'
    m = re.match(r"^(?:(\d+)\s*m\s*)?(\d+(?:\.\d+)?)\s*s$", t)
    if m:
        minutes = float(m.group(1) or 0)
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    # Handle 'm:ss' or 'mm:ss.s'
    if ':' in t:
        mins_str, secs_str = t.split(':', 1)
        # strip non-numeric suffixes like 's'
        secs_str = re.sub(r"[^0-9\.]", "", secs_str)
        try:
            return float(mins_str) * 60 + float(secs_str or 0)
        except ValueError:
            return 0.0
    # Pure seconds possibly with trailing 's'
    t = re.sub(r"[^0-9\.]", "", t)
    try:
        return float(t)
    except ValueError:
        return 0.0


def timestamp_to_frame(timestamp: str, fps: int) -> int:
    return int(parse_timestamp(timestamp) * fps)


def wrap_text(text: str, font, scale: float, thickness: int, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        text_size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        if text_size[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def get_center_of_landmarks(landmarks, width: int, height: int) -> Tuple[int, int]:
    xs = [int(lm.x * width) for lm in landmarks]
    ys = [int(lm.y * height) for lm in landmarks]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def compute_bbox_from_landmarks(landmarks, width: int, height: int) -> Tuple[int, int, int, int]:
    xs = [int(lm.x * width) for lm in landmarks]
    ys = [int(lm.y * height) for lm in landmarks]
    min_x, max_x = max(0, min(xs)), min(width - 1, max(xs))
    min_y, max_y = max(0, min(ys)), min(height - 1, max(ys))
    return min_x, min_y, max(0, max_x - min_x), max(0, max_y - min_y)


def draw_translucent_box(img, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], alpha: float = 0.4) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# -------------------------
# Chord diagram generation
# -------------------------

def normalize_chord_key(label: str) -> str:
    if not label:
        return ''
    s = label.lower().strip()
    s = s.replace('major', 'maj').replace('minor', 'min')
    s = s.replace('–', '-').replace('—', '-')
    s = s.replace('\n', ' ')
    s = ' '.join(s.split())
    # reduce like 'g maj' -> 'g_maj'
    s = s.replace(' ', '_')
    return s


def get_chord_shape(label: str) -> Optional[Dict[str, object]]:
    """Return a chord shape spec for common open chords.

    Spec format:
      {
        'start_fret': 1,
        'strings': [f6, f5, f4, f3, f2, f1],  # -1 muted, 0 open, >0 fret number
        'fingers': [n6..n1],                  # 0 for open/muted, 1..4 finger index
      }
    """
    key = normalize_chord_key(label)
    shapes: Dict[str, Dict[str, object]] = {
        'g_maj': {  # E A D G B e
            'start_fret': 1,
            'strings': [3, 2, 0, 0, 0, 3],
            'fingers': [3, 2, 0, 0, 0, 4],
        },
        'a_min': {
            'start_fret': 1,
            'strings': [-1, 0, 2, 2, 1, 0],
            'fingers': [0, 0, 3, 2, 1, 0],
        },
        'e_min': {
            'start_fret': 1,
            'strings': [0, 2, 2, 0, 0, 0],
            'fingers': [0, 2, 3, 0, 0, 0],
        },
        'd_maj': {
            'start_fret': 1,
            'strings': [-1, -1, 0, 2, 3, 2],
            'fingers': [0, 0, 0, 1, 3, 2],
        },
        'c_maj': {
            'start_fret': 1,
            'strings': [-1, 3, 2, 0, 1, 0],
            'fingers': [0, 3, 2, 0, 1, 0],
        },
    }
    # try exact key first
    if key in shapes:
        return shapes[key]
    # allow just the root letter (e.g., 'g') for a default major
    root = key.split('_')[0] if key else ''
    if root in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
        fallback = f"{root}_maj"
        return shapes.get(fallback)
    return None


def render_chord_diagram(label: str, width: int = 260, height: int = 360) -> Optional[np.ndarray]:
    shape = get_chord_shape(label)
    if not shape:
        return None

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_x = 30
    margin_top = 60
    margin_bottom = 40
    grid_w = width - 2 * margin_x
    grid_h = height - margin_top - margin_bottom

    # geometry
    num_strings = 6
    num_frets = 5
    string_xs = [int(margin_x + i * grid_w / (num_strings - 1)) for i in range(num_strings)]
    fret_ys = [int(margin_top + j * grid_h / num_frets) for j in range(num_frets + 1)]

    # nut or starting fret indicator
    start_fret = int(shape['start_fret'])
    line_color = (0, 0, 0)
    if start_fret == 1:
        cv2.rectangle(img, (margin_x - 6, fret_ys[0] - 6), (margin_x + grid_w + 6, fret_ys[0] + 6), line_color, thickness=-1)
    else:
        cv2.putText(img, f"{start_fret}fr", (margin_x - 20, fret_ys[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2, cv2.LINE_AA)

    # draw strings
    for x in string_xs:
        cv2.line(img, (x, fret_ys[0]), (x, fret_ys[-1]), line_color, 2)
    # draw frets
    for y in fret_ys[1:]:
        cv2.line(img, (margin_x, y), (margin_x + grid_w, y), line_color, 2)

    # draw markers (from low E at index 0 to high e at 5)
    strings = list(shape['strings'])
    fingers = list(shape['fingers'])
    for si, fret in enumerate(strings):
        x = string_xs[si]
        finger = int(fingers[si]) if si < len(fingers) else 0
        if fret < 0:
            # muted: 'X' above nut
            cv2.putText(img, 'X', (x - 10, fret_ys[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        elif fret == 0:
            cv2.circle(img, (x, fret_ys[0] - 16), 10, (0, 0, 0), 2)
        else:
            # position between fret lines
            if fret > num_frets:
                # out of visible range; clamp to last available slot
                fret = num_frets
            y_top = fret_ys[fret - 1]
            y_bot = fret_ys[fret]
            cy = (y_top + y_bot) // 2
            cv2.circle(img, (x, cy), 12, (0, 0, 0), thickness=-1)
            if finger > 0:
                cv2.putText(img, str(finger), (x - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # title at top
    title = label
    tsize = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    tx = max(0, (width - tsize[0]) // 2)
    cv2.putText(img, title, (tx, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    return img


def main() -> None:
    # Load timeline data from JSON
    try:
        with open(config['json_file'], 'r') as f:
            timeline = json.load(f)
        print(f"Successfully loaded {config['json_file']}")
    except Exception as e:
        print(f"Error loading JSON file {config['json_file']}: {e}")
        return

    # Open the video twice (processing/display)
    process_cap = cv2.VideoCapture(config['video_file'])
    if not process_cap.isOpened():
        print(f"Error: Could not open video file {config['video_file']}")
        return
    process_fps = int(process_cap.get(cv2.CAP_PROP_FPS))
    process_width = int(process_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    process_height = int(process_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    display_cap = cv2.VideoCapture(config['video_file'])
    if not display_cap.isOpened():
        print(f"Error: Could not open display video file {config['video_file']}")
        return
    display_fps = int(display_cap.get(cv2.CAP_PROP_FPS))
    display_width = int(display_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    display_height = int(display_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"Video loaded: {config['video_file']} | FPS: {process_fps}, Res: {process_width}x{process_height}"
    )

    # Normalize input JSON to a common structure
    events: List[Dict[str, str]] = []
    raw_key = (
        'feedback_analysis' if 'feedback_analysis' in timeline else
        'events' if 'events' in timeline else
        'chords' if 'chords' in timeline else
        'shots'
    )
    for e in timeline.get(raw_key, []):
        ts = (
            e.get('timestamp')
            or e.get('timestamp_of_outcome')
            or e.get('time')
            or '0:00'
        )
        chord = e.get('chord') or e.get('label') or e.get('shot_type') or 'Unknown'
        # Keep title/accuracy separate for UI
        title = e.get('analysisTitle') or e.get('title')
        base_feedback = (
            e.get('feedback')
            or e.get('technique_feedback')
            or e.get('rhythm_feedback')
            or ''
        )
        accuracy_text = e.get('accuracy') or ''
        acc_num_str = re.sub(r"[^0-9\.]", "", accuracy_text)
        try:
            acc_pct = max(0, min(100, float(acc_num_str))) if acc_num_str else None
        except ValueError:
            acc_pct = None
        events.append({
            'timestamp': ts,
            'chord': chord,
            'title': title or '',
            'feedback': base_feedback,
            'accuracy_text': accuracy_text,
            'accuracy_pct': acc_pct,
        })

    # Map timestamps to frames and set feedback visibility duration
    for ev in events:
        ev['frame_number'] = timestamp_to_frame(ev['timestamp'], process_fps)
        ev['feedback_end_frame'] = ev['frame_number'] + (5 * process_fps)

    # Init MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    processed_frames: List[np.ndarray] = []
    frame_count = 0
    process_every_n_frames = max(1, int(process_fps / 20))

    # UI constants
    white = (255, 255, 255)
    black = (0, 0, 0)

    current_feedback: Optional[str] = None
    current_chord: Optional[str] = None
    last_event_time: Optional[float] = None
    animation_duration_sec = 1.25
    current_color = white

    print(f"Processing guitar video: {config['video_file']} with data from {config['json_file']}...")

    last_left_hand_center: Optional[Tuple[int, int]] = None
    last_right_hand_center: Optional[Tuple[int, int]] = None
    last_left_lm_obj = None
    last_right_lm_obj = None
    current_title: str = ''
    current_accuracy_pct: Optional[float] = None

    while process_cap.isOpened() and display_cap.isOpened():
        process_ret, process_frame = process_cap.read()
        display_ret, display_frame = display_cap.read()
        if not process_ret or not display_ret:
            break

        frame_count += 1

        # Run hand tracking sparsely
        if frame_count % process_every_n_frames == 0:
            rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                left_candidates: List[Tuple[object, float]] = []
                right_candidates: List[Tuple[object, float]] = []
                for lm_obj, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handed.classification[0].label  # 'Left' or 'Right'
                    score = handed.classification[0].score
                    if label.lower().startswith('left'):
                        left_candidates.append((lm_obj, score))
                    else:
                        right_candidates.append((lm_obj, score))

                if left_candidates:
                    last_left_lm_obj, _ = max(left_candidates, key=lambda t: t[1])
                    last_left_hand_center = get_center_of_landmarks(last_left_lm_obj.landmark, display_width, display_height)
                if right_candidates:
                    last_right_lm_obj, _ = max(right_candidates, key=lambda t: t[1])
                    last_right_hand_center = get_center_of_landmarks(last_right_lm_obj.landmark, display_width, display_height)

        # Draw hand skeletons and overlay chord label near the RIGHT hand (strumming hand)
        if last_left_lm_obj is not None:
            mp_drawing.draw_landmarks(display_frame, last_left_lm_obj, mp.solutions.hands.HAND_CONNECTIONS)

        if last_right_lm_obj is not None:
            mp_drawing.draw_landmarks(display_frame, last_right_lm_obj, mp.solutions.hands.HAND_CONNECTIONS)
            # Bounding box for right hand
            bx, by, bw, bh = compute_bbox_from_landmarks(last_right_lm_obj.landmark, display_width, display_height)
            # Label near top of bbox
            label = f"{current_chord or 'Detecting...'}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 3
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            pad_x, pad_y = 16, 12
            box_w = text_size[0] + pad_x * 2
            box_h = text_size[1] + pad_y * 2
            label_x = max(10, min(display_width - box_w - 10, bx))
            label_y = max(10, by - box_h - 12)
            draw_translucent_box(display_frame, label_x, label_y, box_w, box_h, (0, 0, 0), alpha=0.45)
            # Event flash color controls text color
            cv2.putText(
                display_frame,
                label,
                (label_x + pad_x, label_y + box_h - pad_y),
                font,
                font_scale,
                current_color,
                thickness,
                cv2.LINE_AA,
            )

        # Activate events and show chord/feedback
        current_feedback = None
        current_chord = current_chord  # keep previous unless a new event hits
        for ev in events:
            if ev['frame_number'] <= frame_count:
                if ev['frame_number'] == frame_count:
                    last_event_time = time.time()
                    current_chord = ev['chord']
                    current_title = ev.get('title', '')
                    current_accuracy_pct = ev.get('accuracy_pct')
                if ev['frame_number'] <= frame_count <= ev['feedback_end_frame']:
                    current_feedback = ev['feedback']

        # Event flash color (controls chord label text color)
        if last_event_time is not None:
            elapsed = time.time() - last_event_time
            if elapsed < animation_duration_sec:
                prog = elapsed / animation_duration_sec
                # Fade yellow -> white
                r = int(255)
                g = int(255)
                b = int(0 + (255 - 0) * prog)
                current_color = (b, g, r)
            else:
                current_color = white
                last_event_time = None

        # Bottom center: feedback (positioned like pickleball video, with text wrapping)
        if current_feedback:
            feedback_font = cv2.FONT_HERSHEY_SIMPLEX
            feedback_scale = 1.0
            feedback_color = white
            feedback_shadow = black
            feedback_thickness = 2
            feedback_spacing = 32
            
            # Limit feedback length and add intelligent splitting
            max_chars_per_section = 120
            if len(current_feedback) > max_chars_per_section:
                # Split at sentence boundaries first, then by length
                sentences = current_feedback.split('. ')
                if len(sentences) > 1:
                    # Take first sentence or two that fit
                    display_feedback = sentences[0]
                    if len(display_feedback) < max_chars_per_section // 2 and len(sentences) > 1:
                        display_feedback += '. ' + sentences[1]
                    if not display_feedback.endswith('.'):
                        display_feedback += '.'
                else:
                    # No sentence breaks, truncate at word boundary
                    words = current_feedback.split()
                    display_feedback = ''
                    for word in words:
                        if len(display_feedback + ' ' + word) <= max_chars_per_section:
                            display_feedback += (' ' if display_feedback else '') + word
                        else:
                            break
                    if display_feedback != current_feedback:
                        display_feedback += '...'
            else:
                display_feedback = current_feedback
            
            max_width = int(display_width * 0.75)  # slightly narrower for readability
            wrapped = wrap_text(display_feedback, feedback_font, feedback_scale, feedback_thickness, max_width)
            total_h = len(wrapped) * feedback_spacing
            start_y = display_height - 45 - total_h  # closer to bottom edge
            for i, line in enumerate(wrapped):
                text_size = cv2.getTextSize(line, feedback_font, feedback_scale, feedback_thickness)[0]
                x = (display_width - text_size[0]) // 2
                y = start_y + i * feedback_spacing
                # Drop shadow for readability
                cv2.putText(display_frame, line, (x + 2, y + 2), feedback_font, feedback_scale, feedback_shadow, feedback_thickness + 1, cv2.LINE_AA)
                cv2.putText(display_frame, line, (x, y), feedback_font, feedback_scale, feedback_color, feedback_thickness, cv2.LINE_AA)

        # Top-left HUD: timer, chord name, title, accuracy bar, chord diagram image
        hud_x = 20
        hud_y = 24
        hud_w = int(display_width * 0.45)

        # Timer (text with drop shadow, no rectangle)
        elapsed_video_s = frame_count / max(1, display_fps)
        timer_text = f"Time: {elapsed_video_s:.1f}s"
        timer_font = cv2.FONT_HERSHEY_DUPLEX
        timer_scale = 0.7
        timer_th = 2
        tsize = cv2.getTextSize(timer_text, timer_font, timer_scale, timer_th)[0]
        # Drop shadow
        cv2.putText(display_frame, timer_text, (hud_x + 2, hud_y + tsize[1] + 2), timer_font, timer_scale, (0, 0, 0), timer_th, cv2.LINE_AA)
        # Main text
        cv2.putText(display_frame, timer_text, (hud_x, hud_y + tsize[1]), timer_font, timer_scale, (255, 255, 255), timer_th, cv2.LINE_AA)
        block_y = hud_y + tsize[1] + 18

        # Chord name (text with drop shadow, no rectangle)
        chord_label = current_chord or 'Detecting...'
        chord_font = cv2.FONT_HERSHEY_DUPLEX
        chord_scale = 1.4
        chord_th = 3
        csize = cv2.getTextSize(chord_label, chord_font, chord_scale, chord_th)[0]
        # Drop shadow
        cv2.putText(display_frame, chord_label, (hud_x + 3, block_y + csize[1] + 3), chord_font, chord_scale, (0, 0, 0), chord_th, cv2.LINE_AA)
        # Main text
        cv2.putText(display_frame, chord_label, (hud_x, block_y + csize[1]), chord_font, chord_scale, (255, 255, 255), chord_th, cv2.LINE_AA)
        block_y += csize[1] + 26

        # Analysis title (if present, text with drop shadow)
        if current_title:
            # Clean up title by removing question marks and extra formatting
            clean_title = current_title.replace('?', '').strip()
            clean_title = ' '.join(clean_title.split())  # normalize whitespace
            
            title_font = cv2.FONT_HERSHEY_DUPLEX
            title_scale = 0.8
            title_th = 2
            max_title_w = hud_w
            lines = wrap_text(clean_title, title_font, title_scale, title_th, max_title_w)
            ty = block_y
            for line in lines:
                lsize = cv2.getTextSize(line, title_font, title_scale, title_th)[0]
                # Drop shadow
                cv2.putText(display_frame, line, (hud_x + 2, ty + lsize[1] + 2), title_font, title_scale, (0, 0, 0), title_th, cv2.LINE_AA)
                # Main text
                cv2.putText(display_frame, line, (hud_x, ty + lsize[1]), title_font, title_scale, (255, 255, 255), title_th, cv2.LINE_AA)
                ty += lsize[1] + 8
            block_y = ty + 8

        # Accuracy bar (blue styling with text drop shadow)
        if current_accuracy_pct is not None:
            acc_text = f"Accuracy: {int(round(current_accuracy_pct))}%"
            acc_font = cv2.FONT_HERSHEY_DUPLEX
            acc_scale = 0.7
            acc_th = 2
            tsize = cv2.getTextSize(acc_text, acc_font, acc_scale, acc_th)[0]
            bar_x = hud_x
            bar_y = block_y
            bar_w = hud_w
            bar_h = 14
            # accuracy text with drop shadow
            cv2.putText(display_frame, acc_text, (bar_x + 2, bar_y + tsize[1] + 2), acc_font, acc_scale, (0, 0, 0), acc_th, cv2.LINE_AA)
            cv2.putText(display_frame, acc_text, (bar_x, bar_y + tsize[1]), acc_font, acc_scale, (255, 255, 255), acc_th, cv2.LINE_AA)
            # progress bar below
            bar_y2 = bar_y + tsize[1] + 18
            draw_translucent_box(display_frame, bar_x, bar_y2, bar_w, bar_h, (30, 30, 30), 0.7)
            fill_w = int((current_accuracy_pct / 100.0) * bar_w)
            draw_translucent_box(display_frame, bar_x, bar_y2, fill_w, bar_h, (220, 120, 0), 0.9)  # Blue color (BGR format)
            block_y = bar_y2 + bar_h + 10

        # Chord diagram image (optional): looks for files under ./chords/
        def find_chord_image(chord_label_text: str) -> Optional[str]:
            if not chord_label_text:
                return None
            base = chord_label_text.lower().strip()
            base = base.replace('major', 'maj').replace('minor', 'min')
            base = base.replace(' ', '_')
            candidates = [
                f"chords/{base}.png",
                f"chords/{base}.jpg",
                f"chords/{base}.jpeg",
            ]
            # Also try first token only (e.g., 'g')
            first = base.split('_')[0]
            candidates += [
                f"chords/{first}.png",
                f"chords/{first}.jpg",
                f"chords/{first}.jpeg",
            ]
            for p in candidates:
                if os.path.exists(p):
                    return p
            return None

        img_path = find_chord_image(current_chord or '')
        diag = None
        if img_path:
            diag = cv2.imread(img_path)
        if diag is None and current_chord:
            # Fallback: render vector diagram
            diag = render_chord_diagram(current_chord)
        if diag is not None:
            # scale to a nice height
            target_h = 180
            scale = target_h / diag.shape[0]
            target_w = int(diag.shape[1] * scale)
            diag = cv2.resize(diag, (target_w, target_h), interpolation=cv2.INTER_AREA)
            # position to the right of text block, but keep on screen
            img_x = min(display_width - target_w - 10, hud_x + hud_w + 10)
            img_y = 24
            # draw translucent backdrop
            draw_translucent_box(display_frame, img_x - 8, img_y - 8, target_w + 16, target_h + 16, (0, 0, 0), 0.35)
            # paste image
            roi = display_frame[img_y:img_y + target_h, img_x:img_x + target_w]
            if roi.shape[:2] == diag.shape[:2]:
                blended = cv2.addWeighted(roi, 0.2, diag, 0.8, 0)
                display_frame[img_y:img_y + target_h, img_x:img_x + target_w] = blended

        processed_frames.append(display_frame.copy())

        cv2.imshow('Guitar Chord Analysis', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    process_cap.release()
    display_cap.release()
    cv2.destroyAllWindows()

    print("Creating final guitar video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = config['output_file']
    out = cv2.VideoWriter(out_path, fourcc, display_fps, (display_width, display_height))
    for frame in processed_frames:
        out.write(frame)
    out.release()
    print(f"Processing complete. Final video saved to {out_path}")


if __name__ == '__main__':
    main()


