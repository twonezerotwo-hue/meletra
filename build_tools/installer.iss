; Inno Setup 6 script. Built by scripts/build_installer.py, which passes the /D defines below.
;   AppVersion  SourceDir (dist\v4.1.0\MELETRA)  OutputDir  OutputBase  IconFile
#ifndef AppVersion
  #define AppVersion "0.1.1"
#endif
#ifndef NoticeFile
  #define NoticeFile "installer_notice.txt"
#endif
#ifndef OutputBase
  #define OutputBase "MELETRA-Setup-Windows-x64"
#endif
#define AppName "MELETRA"
#define AppExe "MELETRA.exe"
#define LegacyExe "MedicalLectureTranscriber.exe"
#define Publisher "Touche Holding"

[Setup]
AppId={{6F0B7C52-3A1E-4E7D-9C54-8D2A7B1E4F60}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#Publisher}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#Publisher}
VersionInfoDescription={#AppName} setup
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; an upgrade from an earlier product name must not keep putting the shortcuts into the Start menu folder of that old name
UsePreviousGroup=no
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBase}
SetupIconFile={#IconFile}
#ifdef WizardLarge
WizardImageFile={#WizardLarge}
#endif
#ifdef WizardSmall
WizardSmallImageFile={#WizardSmall}
#endif
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/normal
SolidCompression=no
LZMANumBlockThreads=8
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
DisableWelcomePage=no
InfoBeforeFile={#NoticeFile}
SetupLogging=yes

[Messages]
; Short, plain copy on the wizard pages: the product name, what it is, nothing about how it is built.
WelcomeLabel1=Welcome to {#AppName}
WelcomeLabel2=Medical Learning & Transcription.%n%nSetup will install {#AppName} {#AppVersion} on your computer. If WhisperMedical or Medical Lecture Transcriber is installed, it is replaced and your lectures and settings are kept. Please close other programs first.
WizardInfoBefore=Your privacy
InfoBeforeLabel=Everything runs on your computer. Please read how sharing works before you continue.
InfoBeforeClickLabel=When you have read it, click Next.
FinishedHeadingLabel={#AppName} is ready
FinishedLabel={#AppName} is installed. Start it from the Start menu or from the desktop shortcut.
FinishedLabelNoIcons={#AppName} is installed. Start it from the Start menu.

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[InstallDelete]
; a build stamp of an earlier version would otherwise stay beside the new program and name the wrong commit
Type: files; Name: "{app}\build_info.json"
; the program was called WhisperMedical, before that Medical Lecture Transcriber: its old program file and shortcuts do not stay beside MELETRA
Type: files; Name: "{app}\{#LegacyExe}"
Type: filesandordirs; Name: "{autoprograms}\WhisperMedical"
Type: filesandordirs; Name: "{autoprograms}\Medical Lecture Transcriber"
Type: files; Name: "{autodesktop}\WhisperMedical.lnk"
Type: files; Name: "{autodesktop}\Medical Lecture Transcriber.lnk"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
const
  UninstallKey = '{6F0B7C52-3A1E-4E7D-9C54-8D2A7B1E4F60}_is1';

// What an earlier installation of this program (same AppId, whatever it was called) recorded: the folder and its uninstaller. Looked up where THIS Setup installs
// (all users: HKLM, this user only: HKCU), so a per-user Setup never touches a machine-wide installation.
function PreviousValue(Name: String): String;
var
  Root: Integer;
  Value: String;
begin
  Result := '';
  if IsAdminInstallMode then Root := HKLM else Root := HKCU;
  if RegQueryStringValue(Root, 'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\' + UninstallKey, Name, Value) then
    Result := Value;
end;

function IsLegacyFolder(Dir: String): Boolean;
var
  Leaf: String;
begin
  Leaf := ExtractFileName(RemoveBackslashUnlessRoot(Dir));
  Result := (CompareText(Leaf, 'WhisperMedical') = 0) or (CompareText(Leaf, 'Medical Lecture Transcriber') = 0);
end;

// An upgrade from a folder that carries the old name goes to the folder of the new name (the old one is removed by its own uninstaller in PrepareToInstall).
procedure InitializeWizard();
var
  Previous: String;
begin
  Previous := PreviousValue('InstallLocation');
  if (Previous <> '') and IsLegacyFolder(Previous) then
    WizardForm.DirEdit.Text := ExpandConstant('{autopf}\{#AppName}');
end;

// The "Installing" page names every file it writes; the names of the program's inner files mean nothing to the person installing, so only the bar and the status show.
procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpInstalling then
    WizardForm.FilenameLabel.Visible := False;
end;

// A running MELETRA.exe (or the program file of an earlier version) locks its own files ("DeleteFile failed; code 5"). Before anything is replaced or removed, Setup
// and the uninstaller ask THIS program (the exe in the install folder, nothing else) to close like a user would, wait for it to exit and
// otherwise stop WITHOUT touching a single file. Other programs (Python, other copies of the app) are never touched.
function RunningScript(Action: String): String;
var
  Dirs, Previous: String;
begin
  Dirs := '''' + ExpandConstant('{app}') + '''';
  Previous := RemoveBackslashUnlessRoot(PreviousValue('InstallLocation'));
  if Previous <> '' then Dirs := Dirs + ',''' + Previous + '''';
  Result := '-NoProfile -NonInteractive -Command "$d = @(' + Dirs + '); $p = @(Get-Process -Name MELETRA,MedicalLectureTranscriber -ErrorAction SilentlyContinue | ' +
            'Where-Object { $d -contains (Split-Path -Parent $_.Path) }); ' + Action + '"';
end;

function AppIsRunning(): Boolean;
var
  Code: Integer;
begin
  Result := Exec('powershell.exe', RunningScript('if ($p.Count -gt 0) { exit 1 } else { exit 0 }'), '', SW_HIDE, ewWaitUntilTerminated, Code)
            and (Code = 1);
end;

procedure AskAppToClose();
var
  Code: Integer;
begin
  Exec('powershell.exe', RunningScript('$p | ForEach-Object { [void]$_.CloseMainWindow() }'), '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

function WaitForAppExit(Seconds: Integer): Boolean;
var
  I: Integer;
begin
  Result := not AppIsRunning();
  I := 0;
  while (not Result) and (I < Seconds) do
  begin
    Sleep(1000);
    I := I + 1;
    Result := not AppIsRunning();
  end;
end;

// True when the program is not running (any more). In a silent run there is nobody to ask, so a program that stays open stops the run.
function EnsureAppClosed(Silent: Boolean): Boolean;
begin
  Result := not AppIsRunning();
  if Result then Exit;
  AskAppToClose();
  Result := WaitForAppExit(25);
  while (not Result) and (not Silent) do
  begin
    if MsgBox('{#AppName} is still open. Close it (finish or cancel a running job first), then click Retry.', mbError, MB_RETRYCANCEL) <> IDRETRY then
      Exit;
    Result := WaitForAppExit(5);
  end;
end;

// An installation in another folder (the old name, an old folder) is removed with its own uninstaller before the new one is written; that uninstaller never deletes the
// user's lectures, settings and models (they are in %LOCALAPPDATA%, and a silent uninstall keeps them). The same folder is simply updated.
function RemovePreviousInstallation(): String;
var
  Previous, Uninstaller: String;
  Code, Waited: Integer;
begin
  Result := '';
  Previous := RemoveBackslashUnlessRoot(PreviousValue('InstallLocation'));
  Uninstaller := RemoveQuotes(PreviousValue('UninstallString'));
  if (Previous = '') or (Uninstaller = '') or (not FileExists(Uninstaller)) then Exit;
  if CompareText(Previous, RemoveBackslashUnlessRoot(ExpandConstant('{app}'))) = 0 then Exit;
  if (not Exec(Uninstaller, '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART', '', SW_HIDE, ewWaitUntilTerminated, Code)) or (Code <> 0) then
  begin
    Result := 'The earlier installation in ' + Previous + ' could not be removed. Nothing was changed; remove it in Settings, Apps, then run Setup again.';
    Exit;
  end;
  Waited := 0;
  while FileExists(Uninstaller) and (Waited < 180) do
  begin
    Sleep(1000);
    Waited := Waited + 1;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not EnsureAppClosed(WizardSilent) then
  begin
    Result := '{#AppName} is still running, so its files cannot be replaced safely. Close it and run Setup again. Nothing was changed.';
    Exit;
  end;
  Result := RemovePreviousInstallation();
end;

function InitializeUninstall(): Boolean;
begin
  Result := EnsureAppClosed(UninstallSilent);
  if (not Result) and (not UninstallSilent) then
    MsgBox('{#AppName} is still running. Close it and run the uninstaller again. Nothing was removed.', mbError, MB_OK);
end;

// Downloaded speech files, settings, logs and unfinished work live in %LOCALAPPDATA%\MedicalLectureTranscriber. They are KEPT by default so a
// re-install or an update never forces a 3 GB model download again.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\MedicalLectureTranscriber');
    if (not UninstallSilent) and DirExists(DataDir) then
      if MsgBox('Also delete your settings, downloaded files and unfinished work?' + #13#10 + DataDir + #13#10#13#10 +
                'Choose No to keep them (recommended if you may install the program again).',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
