<#
.SYNOPSIS
    Renames a lecture through the REAL window of the installed program with Windows UI Automation: the card's "Actions for ..." button, the menu entry "Rename", the in-place editor, Enter.
.DESCRIPTION
    Used by scripts/installed_local_check.py on a COPY of the profile. The window is brought to the front and the keyboard is used only after the window really is the foreground window
    (otherwise the script stops without sending a key). Output: one line "RENAMED <new name>" or "FAILED <reason>".
#>
param(
    [Parameter(Mandatory = $true)][Int64]$Hwnd,
    [Parameter(Mandatory = $true)][string]$NewName,
    [string]$Match = 'Actions for *',                                 # which card: the name of its "Actions for ..." button
    [string]$Page = ''                                                # a sidebar page to open first (for example Files: the archive of every lecture)
)

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type -Namespace Win -Name Native -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern bool SetForegroundWindow(System.IntPtr hWnd);
[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern System.IntPtr GetForegroundWindow();
[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern bool ShowWindow(System.IntPtr hWnd, int nCmdShow);
'@

function Fail($reason) { Write-Output "FAILED $reason"; exit 1 }
function Find-All($scope, $type, $nameLike) {
    $cond = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ControlTypeProperty, $type)
    $all = $scope.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    @($all | Where-Object { $_.Current.Name -like $nameLike })
}

$window = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$Hwnd)
[Win.Native]::ShowWindow([IntPtr]$Hwnd, 9) | Out-Null
[Win.Native]::SetForegroundWindow([IntPtr]$Hwnd) | Out-Null
Start-Sleep -Milliseconds 800
if ([Win.Native]::GetForegroundWindow() -ne [IntPtr]$Hwnd) { Fail "the program window is not the foreground window (no key was sent)" }

if ($Page) {
    $nav = (Find-All $window ([System.Windows.Automation.ControlType]::Button) $Page) | Select-Object -First 1
    if (-not $nav) { $nav = (Find-All $window ([System.Windows.Automation.ControlType]::CheckBox) $Page) | Select-Object -First 1 }      # a checkable button reads as a check box
    if (-not $nav) { Fail "no sidebar button $Page" }
    $pattern = $null
    if ($nav.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$pattern)) { $pattern.Invoke() }
    elseif ($nav.TryGetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern, [ref]$pattern)) { $pattern.Toggle() }
    else { Fail "the sidebar button $Page cannot be pressed through UI Automation" }
    Start-Sleep -Milliseconds 1500
}
$button = (Find-All $window ([System.Windows.Automation.ControlType]::Button) $Match) | Select-Object -First 1
if (-not $button) { Fail "no card menu button found" }
$old = $button.Current.Name -replace '^Actions for ', ''
$button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
Start-Sleep -Milliseconds 800

$desktop = [System.Windows.Automation.AutomationElement]::RootElement
$item = $null
for ($i = 0; $i -lt 10 -and -not $item; $i++) {
    $item = (Find-All $desktop ([System.Windows.Automation.ControlType]::MenuItem) 'Rename*') | Select-Object -First 1
    if (-not $item) { Start-Sleep -Milliseconds 300 }
}
if (-not $item) { Fail "no 'Rename' menu entry appeared" }
$item.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
Start-Sleep -Milliseconds 800
# a menu entry invoked through UI Automation runs its action but leaves the pop-up open (a mouse click would close it): the pop-up still holds the keyboard, so it is closed first
if ((Find-All $desktop ([System.Windows.Automation.ControlType]::MenuItem) 'Rename*').Count -gt 0) {
    [System.Windows.Forms.SendKeys]::SendWait('{ESC}')
    Start-Sleep -Milliseconds 800
}

$edit = $null
for ($i = 0; $i -lt 10 -and -not $edit; $i++) {
    $edit = (Find-All $window ([System.Windows.Automation.ControlType]::Edit) 'Lecture name*') | Where-Object { -not $_.Current.IsOffscreen } | Select-Object -First 1
    if (-not $edit) { Start-Sleep -Milliseconds 300 }
}
if (-not $edit) { Fail "the in-place name editor did not open" }
# the card has already focused the editor and selected its text: typing replaces it (leaving the field would cancel the rename, so nothing else touches the focus)
Start-Sleep -Milliseconds 1500                                   # the menu closes and the editor takes the focus; keys sent too early are lost
if ([Win.Native]::GetForegroundWindow() -ne [IntPtr]$Hwnd) { Fail "the program window lost the foreground before typing (no key was sent)" }
foreach ($ch in $NewName.ToCharArray()) { [System.Windows.Forms.SendKeys]::SendWait(([string]$ch -replace '[+^%~(){}\[\]]', '{$0}')); Start-Sleep -Milliseconds 40 }          # one key at a time
Start-Sleep -Milliseconds 300
[System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
Start-Sleep -Milliseconds 800
Write-Output "RENAMED $old -> $NewName"
