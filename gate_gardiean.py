
import cv2
import numpy as np
import argparse
import time
import os
import csv
from datetime import datetime
import smtplib
from email.message import EmailMessage
import sys
import threading

# Optional RPi GPIO — used only if available and user enables --use-gpio
try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except Exception:
    RPI_AVAILABLE = False

# ------------- Configuration & CLI -------------
parser = argparse.ArgumentParser(description="Gate Guardian - simple camera-based gate monitor")
parser.add_argument("--source", type=str, default="0",
                    help="Camera source. 0 (default) for webcam, or a video stream URL (rtsp/http/file).")
parser.add_argument("--min-area", type=int, default=500, help="Minimum contour area to consider motion (px)")
parser.add_argument("--snapshot-dir", "--out", dest="outdir", default="snapshots",
                    help="Directory to save detection snapshots.")
parser.add_argument("--log", dest="logfile", default="events_log.csv", help="CSV file for event logging.")
parser.add_argument("--display", action="store_true", help="Show video window (default off).")
parser.add_argument("--headless", action="store_true", help="Run without GUI display (useful on servers).")
parser.add_argument("--gate-pin", type=int, default=None, help="GPIO pin to control gate (RPi).")
parser.add_argument("--gate-simulate", action="store_true", help="Simulate gate open/close in logs even without GPIO.")
parser.add_argument("--email-alert", action="store_true", help="Send email on detection (configure env below or edit function).")
parser.add_argument("--email-to", type=str, default="", help="Recipient email for alerts.")
parser.add_argument("--sensitivity", type=float, default=1.2, help="HOG detection scale factor (lower -> more sensitive).")
parser.add_argument("--cooldown", type=int, default=15, help="Cooldown seconds between repeated alerts.")
parser.add_argument("--fps-limit", type=float, default=10.0, help="Max processing FPS (to reduce CPU).")
args = parser.parse_args()

# Interpret camera source
if args.source.isdigit():
    cam_index = int(args.source)
    source = cam_index
else:
    source = args.source

os.makedirs(args.outdir, exist_ok=True)

# ------------- Logging utility -------------
def init_log(path):
    existed = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not existed:
        writer.writerow(["timestamp", "event", "details"])
        f.flush()
    return f, writer

logfile_handle, log_writer = init_log(args.logfile)

def log_event(event, details=""):
    ts = datetime.utcnow().isoformat() + "Z"
    log_writer.writerow([ts, event, details])
    logfile_handle.flush()
    print(f"[{ts}] {event} {details}")

# ------------- Email alert (simple) -------------
# NOTE: For real deployment, use environment variables, app passwords, or an email API.
SMTP_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "",   # set here if not using envs
    "smtp_pass": "",   # set here if not using envs
}

def send_email_alert(subject, body, to_addr):
    if not to_addr:
        log_event("email_failed", "No recipient set")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_CONFIG["smtp_user"]
        msg["To"] = to_addr
        msg.set_content(body)

        with smtplib.SMTP(SMTP_CONFIG["smtp_server"], SMTP_CONFIG["smtp_port"]) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_CONFIG["smtp_user"], SMTP_CONFIG["smtp_pass"])
            s.send_message(msg)
        log_event("email_sent", f"to={to_addr} subject={subject}")
        return True
    except Exception as e:
        log_event("email_error", str(e))
        return False

# ------------- Gate control (GPIO or simulation) -------------
gate_state = "closed"
last_gate_action_time = 0

def setup_gpio(pin):
    if not RPI_AVAILABLE:
        log_event("gpio_unavailable", "RPi.GPIO not installed")
        return False
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
        log_event("gpio_setup", f"pin={pin}")
        return True
    except Exception as e:
        log_event("gpio_setup_error", str(e))
        return False

def trigger_gate(duration=3):
    global gate_state, last_gate_action_time
    last_gate_action_time = time.time()
    if args.gate_pin is not None and RPI_AVAILABLE:
        try:
            log_event("gate_trigger", f"gpio_pin={args.gate_pin}")
            GPIO.output(args.gate_pin, GPIO.HIGH)
            gate_state = "opening"
            time.sleep(duration)
            GPIO.output(args.gate_pin, GPIO.LOW)
            gate_state = "closed"
            log_event("gate_action", "pulse completed")
            return
        except Exception as e:
            log_event("gpio_error", str(e))
    # Simulation fallback
    if args.gate_simulate:
        log_event("gate_simulated", f"simulate_duration={duration}s")
        gate_state = "opening"
        time.sleep(duration)
        gate_state = "closed"
    else:
        log_event("gate_noaction", "No GPIO and simulation disabled")

# ------------- Detector setup -------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people(frame, scale=args.sensitivity):
    # frame: color BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detectMultiScale returns rects
    rects, weights = hog.detectMultiScale(gray, winStride=(8,8), padding=(16,16), scale=scale)
    # Filter by aspect/size heuristics if needed
    bboxes = []
    for (x, y, w, h) in rects:
        if w*h < 500:  # ignore tiny
            continue
        bboxes.append((x, y, w, h))
    return bboxes, weights

# ------------- Motion detection helper -------------
class MotionDetector:
    def __init__(self, min_area=500):
        self.min_area = min_area
        self.avg = None

    def detect(self, frame):
        # frame must be grayscale
        if self.avg is None:
            self.avg = frame.astype("float")
            return []
        # accumulate weighted average
        cv2.accumulateWeighted(frame, self.avg, 0.5)
        frameDelta = cv2.absdiff(frame, cv2.convertScaleAbs(self.avg))
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate and find contours
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append((x,y,w,h))
        return boxes

# ------------- Main loop -------------
def main():
    # Setup video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log_event("camera_error", f"cannot open source {source}")
        print("ERROR: cannot open camera/source. Exiting.")
        sys.exit(1)
    log_event("camera_opened", f"source={source}")

    md = MotionDetector(min_area=args.min_area)
    last_alert_time = 0
    last_processed = 0.0
    frame_count = 0

    # Optionally setup GPIO
    if args.gate_pin is not None:
        if setup_gpio(args.gate_pin):
            log_event("gpio_ready", f"pin={args.gate_pin}")
        else:
            log_event("gpio_failed_setup", f"pin={args.gate_pin}")

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                log_event("frame_error", "empty frame")
                time.sleep(0.5)
                continue

            # Optionally limit FPS
            elapsed_since_last = start - last_processed
            if args.fps_limit > 0:
                min_frame_time = 1.0 / args.fps_limit
                if elapsed_since_last < min_frame_time:
                    time.sleep(max(0, min_frame_time - elapsed_since_last))
            last_processed = time.time()

            # Resize for faster processing
            h0, w0 = frame.shape[:2]
            max_dim = 800
            if max(h0, w0) > max_dim:
                scale = max_dim / float(max(h0, w0))
                frame = cv2.resize(frame, (int(w0*scale), int(h0*scale)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_boxes = md.detect(gray)
            people_boxes, weights = detect_people(frame)

            # Merge motion + people detection heuristics:
            # We'll only treat a detection as person if person bbox overlaps motion bbox or at least a person bbox exists
            detected_people = []
            for (x,y,w,h) in people_boxes:
                # check if overlaps any motion box (optional)
                overlaps = any(overlap_rect((x,y,w,h), mb) for mb in motion_boxes) if motion_boxes else True
                if overlaps:
                    detected_people.append((x,y,w,h))

            event_triggered = False
            details = ""
            if detected_people:
                event_triggered = True
                details = f"people={len(detected_people)}"
            elif motion_boxes:
                # motion but no person detected — flag as suspicious
                event_triggered = True
                details = f"motion={len(motion_boxes)}"
            # handle events
            now = time.time()
            if event_triggered and (now - last_alert_time) > args.cooldown:
                last_alert_time = now
                tsstr = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                snapshot_path = os.path.join(args.outdir, f"detection_{tsstr}.jpg")
                # annotate frame before saving
                aframe = frame.copy()
                for (x,y,w,h) in detected_people:
                    cv2.rectangle(aframe, (x,y), (x+w, y+h), (0,255,0), 2)
                for (x,y,w,h) in motion_boxes:
                    cv2.rectangle(aframe, (x,y), (x+w, y+h), (0,0,255), 1)
                cv2.putText(aframe, f"{details}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imwrite(snapshot_path, aframe)
                log_event("detection", f"{details} snapshot={snapshot_path}")

                # trigger gate (non-blocking)
                threading.Thread(target=trigger_gate, kwargs={"duration":3}, daemon=True).start()

                # email alert (non-blocking)
                if args.email_alert:
                    subject = f"Gate Guardian Alert: {details}"
                    body = f"Alert at {datetime.utcnow().isoformat()}Z\nDetails: {details}\nSnapshot: {snapshot_path}"
                    threading.Thread(target=send_email_alert, args=(subject, body, args.email_to), daemon=True).start()

            # Display if requested and not headless
            if args.display and not args.headless:
                draw = frame.copy()
                for (x,y,w,h) in detected_people:
                    cv2.rectangle(draw, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(draw, "Person", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                for (x,y,w,h) in motion_boxes:
                    cv2.rectangle(draw, (x,y), (x+w, y+h), (0,0,255), 1)
                status_text = f"Gate:{gate_state} | LastAction:{int(time.time()-last_gate_action_time)}s"
                cv2.putText(draw, status_text, (10, draw.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow("Gate Guardian", draw)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_count += 1

    except KeyboardInterrupt:
        log_event("stopped", "keyboard interrupt")
    finally:
        cap.release()
        if args.display and not args.headless:
            cv2.destroyAllWindows()
        logfile_handle.close()
        if args.gate_pin is not None and RPI_AVAILABLE:
            GPIO.cleanup()

# ------------- Utilities -------------
def overlap_rect(a, b):
    # a and b are (x,y,w,h)
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1+aw, ay1+ah
    bx2, by2 = bx1+bw, by1+bh
    # overlap
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter_area = (ix2-ix1) * (iy2-iy1)
    a_area = aw*ah
    b_area = bw*bh
    # consider overlap meaningful if > 10% of either box
    if inter_area / float(a_area+1e-6) > 0.1 or inter_area / float(b_area+1e-6) > 0.1:
        return True
    return False

# ------------- Run -------------
if __name__ == "__main__":
    # quick sanity message / config hint
    log_event("startup", f"source={args.source} headless={args.headless} display={args.display}")
    if args.email_alert and not SMTP_CONFIG["smtp_user"]:
        log_event("email_disabled", "SMTP not configured in script; edit SMTP_CONFIG or set envs.")
    main()
