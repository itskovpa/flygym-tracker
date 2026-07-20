; Windows installer for FlyGym Tracker.
;
; WHAT THIS SHIPS: the PyInstaller folder from `flygym_tracker.spec` -- the program, a private
; Python, PySide6, OpenCV, NumPy, pandas and the config templates. A machine that has never had
; Python on it can run this.
;
; WHAT IT CANNOT SHIP, and says so instead: the HikRobot MVS SDK. It carries USB3 Vision DRIVERS,
; which cannot be delivered by copying files, and its licence is not ours to redistribute. So the
; installer CHECKS for it and, if it is missing, tells the operator plainly -- rather than letting
; them find out when the camera will not open on the morning they load flies.
;
; PER-USER INSTALL BY DEFAULT (`PrivilegesRequiredOverridesAllowed=dialog`): a lab machine is
; frequently one a scientist cannot install software on. Asking for admin unconditionally turns
; "install the tracker" into "find whoever administers this computer".

#define AppName        "FlyGym Tracker"
#define AppPublisher   "Pavel Itskov"
#define AppExeName     "FlyGymTracker.exe"
#define AppURL         "https://github.com/itskovpa/flygym-tracker"

; Passed in by build_installer.py so the version lives in exactly one place.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "dist\FlyGymTracker"
#endif
#ifndef OutputDir
  #define OutputDir "dist"
#endif

[Setup]
AppId={{7A3B1C42-9E55-4C1D-9A6E-2F5D8B0C1E37}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename=FlyGymTracker-{#AppVersion}-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; 64-bit only: the bundled Python, Qt and OpenCV are all x64.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
; ~290 MB of payload; saying so up front avoids a customer aborting a download they thought was small.
DiskSpanning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\INSTALL.md"; DestDir: "{app}"; DestName: "INSTALL.txt"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

; NOTHING IN [UninstallDelete]. The operator's data -- settings, vial positions, and the RESULTS OF
; THEIR EXPERIMENTS -- lives in Documents\FlyGym Tracker, and an uninstaller that deleted a folder
; of research data because somebody removed a program would be indefensible. Uninstalling removes
; the program and leaves the science.

[Code]
function MvsIsInstalled(): Boolean;
{ The MVS SDK's Python bindings are what `frame_source.py` actually imports. Checking for the
  folder it imports FROM is the honest test -- a registry key would prove only that something
  called MVS was once installed. }
begin
  Result := DirExists(ExpandConstant('{commonpf32}\MVS\Development\Samples\Python\MvImport')) or
            DirExists(ExpandConstant('{commonpf}\MVS\Development\Samples\Python\MvImport'));
end;

function InitializeSetup(): Boolean;
var
  Message: String;
begin
  Result := True;
  if not MvsIsInstalled() then
  begin
    Message :=
      'The HikRobot MVS software was not found on this computer.' + #13#10#13#10 +
      '{#AppName} needs it to talk to the camera. It is a separate free download from HikRobot ' +
      '(MVS 4.8 or newer), and it carries the USB3 Vision drivers, so it cannot be included in ' +
      'this installer.' + #13#10#13#10 +
      'You can install {#AppName} now and add MVS later -- everything except the live camera ' +
      'works without it, including replaying a recording and reading results.' + #13#10#13#10 +
      'Continue with the installation?';
    { Default to Yes: the operator may well be setting up in an order of their own choosing, and
      an installer that refuses outright is one they cannot get past at all. }
    Result := MsgBox(Message, mbConfirmation, MB_YESNO or MB_DEFBUTTON1) = IDYES;
  end;
end;
