# Installing FlyGym Tracker

FlyGym Tracker measures the locomotor activity of *Drosophila* in the 32 vials of a FlyGym v2
rotating drum, one row of numbers per vial per time bin, for experiments lasting hours to days.

You need **two** things on the computer. The installer is one of them; the camera software is the
other, and it cannot be included — see below.

---

## 1. FlyGym Tracker

Run **`FlyGymTracker-<version>-Setup.exe`** and follow the prompts.

Everything the program needs is inside it — including its own copy of Python. **You do not need to
install Python**, or anything from PyPI, and nothing already on the machine is changed.

- Installs by default to your own account, so **you do not need administrator rights**. (The
  installer offers a machine-wide install if you have them and want one.)
- About **700 MB** of disk once installed.
- Windows 10 or 11, 64-bit.

**Windows SmartScreen will probably warn you** the first time you run it, saying the publisher is
unrecognised. That is expected: the installer is not code-signed (a signing certificate is an
annual commercial expense). Click *More info* → *Run anyway*. If you would rather verify the file
first, check its SHA-256 against the checksum published with the release.

## 2. HikRobot MVS — required for the camera

Download and install **MVS 4.8 or newer** from HikRobot's own site (search "HikRobot MVS
download"; it is a free download, and the camera is likely to have shipped with a copy).

**Why this is not in the installer.** MVS contains the USB3 Vision *drivers* the camera needs.
Drivers cannot be installed by copying files, and HikRobot's licence does not permit us to
redistribute their SDK. So it stays a separate download — the installer checks whether it is there
and tells you if it is not.

FlyGym Tracker installs and runs perfectly well without MVS. Everything except the live camera
works: replaying a recording, re-reading results, drawing vial positions on a saved frame. Only
opening the camera needs it.

> **The camera allows one program at a time.** USB3 Vision access is exclusive. If the MVS Viewer
> is open, FlyGym Tracker cannot open the camera, and vice versa. Close one before starting the
> other.

---

## Where your files go

The program is installed in one place; **everything you produce is kept somewhere else** — under
your own Documents folder, in `FlyGym Tracker`:

```
Documents\FlyGym Tracker\
    config\flygym_rig.local.yaml   this rig's settings (yours; safe to edit or delete)
    calib_faces\                   the vial positions you drew
    output\                        results: activity CSVs, the Excel workbook, run metadata
```

They are deliberately **not** inside the installation folder. Windows does not let a normal user
write into `C:\Program Files`, and — worse — it does not always say so: it silently redirects the
write to a hidden per-user copy, so the program looks like it saved and the files are not where
anyone would look for them. That failure would land on a three-day experiment.

**Uninstalling does not delete this folder.** Removing a program should never delete research data.
If you want the results gone, delete the folder yourself.

To put your data somewhere else — a data drive, or a shared folder on a rig several people use —
set the environment variable `FLYGYM_DATA_DIR` to the folder you want.

---

## Checking it worked

1. Start **FlyGym Tracker** from the Start menu.
2. The window should open with the picture area on top and a settings panel to the left.
3. Press **Open camera**. With MVS installed and the camera plugged in, you should see live frames
   and a measured frame rate in the top bar. If it says the camera is busy, something else has it
   — close MVS Viewer.
4. Without a camera, use **Replay a recording** to run the identical analysis over a saved video.
   That is the quickest way to confirm the analysis half of the program works end to end.

If the program does not start at all, it writes `startup-error.log` into your
`Documents\FlyGym Tracker` folder — that file names the cause.

---

## Upgrading

Run the new installer over the old one. Your settings, vial positions and results are untouched:
they live outside the installation folder.
