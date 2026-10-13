"""Fill the worklist with real X-ray images for a demo (live presentation).

Each image is analyzed by the real model and added as a case. Arrival times are
spread out (one case every few minutes, in random order), so the FIFO and AI
orders are clearly different on the Worklist page.

Run from the repository root:
    python app/seed_demo.py "C:\\path\\to\\demo_images"
    python app/seed_demo.py "C:\\path\\to\\demo_images" --n 20 --minutes 3 --reset

The folder can have class sub-folders (COVID, Lung_Opacity, Normal, Viral Pneumonia).
Then the true class is printed next to the prediction (it is NOT stored as the
radiologist label: the radiologist decides in the app).
Use test-set images only, never training images.
"""
import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db
from inference import InvalidImageError, analyze, load_image

EXTENSIONS = {".png", ".jpg", ".jpeg"}


def true_class(path, folder):
    """Class name from a sub-folder name, if there is one."""
    for part in path.relative_to(folder).parts[:-1]:
        for cls in db.CLASSES:
            if part.lower().replace(" ", "_") == cls.lower().replace(" ", "_"):
                return cls
    return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="Folder with demo X-ray images")
    ap.add_argument("--n", type=int, default=20, help="Number of images to use (default 20)")
    ap.add_argument("--minutes", type=float, default=3,
                    help="Minutes between two arrivals (default 3)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for the order")
    ap.add_argument("--reset", action="store_true",
                    help="Delete ALL existing cases first (asks for confirmation)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        sys.exit(f"No PNG/JPG images found in {folder}")

    random.seed(args.seed)
    random.shuffle(files)          # random arrival order, like a real department
    files = files[:args.n]

    db.init_db()
    if args.reset:
        answer = input("Delete ALL existing cases and images? Type yes: ")
        if answer.strip().lower() != "yes":
            sys.exit("Cancelled. Nothing was changed.")
        db.reset_all()
        print("Old cases deleted.")

    # Arrivals in the past: first case n*minutes ago, last case about now
    start = datetime.now().astimezone() - timedelta(minutes=args.minutes * len(files))
    print(f"Adding {len(files)} cases (model loads on the first image, please wait)...\n")
    print(f"{'#':>3}  {'arrival':8}  {'true':16} {'predicted':16} {'conf':>6}  {'urgency':>7}")

    added, correct, known = 0, 0, 0
    for i, path in enumerate(files):
        try:
            image = load_image(path.name, path.read_bytes())
        except InvalidImageError as e:
            print(f"  skipped {path.name}: {e}")
            continue
        res = analyze(image)
        arrival = start + timedelta(minutes=args.minutes * i)
        db.save_case(image, res, uploaded_at=arrival.isoformat(timespec="seconds"))
        added += 1

        truth = true_class(path, folder)
        if truth != "?":
            known += 1
            correct += truth == res["class"]
        print(f"{added:>3}  {arrival:%H:%M}     {truth:16} {res['class']:16} "
              f"{res['confidence']:6.1%}  {db.urgency_score(res['probabilities']):7.2f}")

    print(f"\nDone: {added} cases added.")
    if known:
        print(f"Model agreed with the true class on {correct}/{known} images.")
    print("Open the app:  streamlit run app/app.py  -> Worklist -> Compare FIFO vs AI")


if __name__ == "__main__":
    main()
