<#
.SYNOPSIS
    Reads the UI Automation tree of a window the way an assistive technology does (Windows UI Automation client: names, control types, keyboard focus, help text).
.DESCRIPTION
    This is a real UIA client talking to the real window through Windows, so it exercises the accessibility bridge of the packaged program. It is NOT a screen reader
    (NVDA, JAWS, Narrator) and says nothing about how one of them speaks the page.
    Output: a JSON array of {name, type, focusable, enabled, help, class, automationId} for every descendant, to the file given with -Out.
#>
param(
    [Parameter(Mandatory = $true)][Int64]$Hwnd,
    [Parameter(Mandatory = $true)][string]$Out
)

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

$root = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$Hwnd)
$elements = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)
$rows = New-Object System.Collections.Generic.List[object]
$rows.Add([pscustomobject]@{
    name = $root.Current.Name; type = $root.Current.ControlType.ProgrammaticName; focusable = $root.Current.IsKeyboardFocusable
    enabled = $root.Current.IsEnabled; help = $root.Current.HelpText; class = $root.Current.ClassName; automationId = $root.Current.AutomationId })
foreach ($e in $elements) {
    try {
        $c = $e.Current
        $rows.Add([pscustomobject]@{
            name = $c.Name; type = $c.ControlType.ProgrammaticName; focusable = $c.IsKeyboardFocusable
            enabled = $c.IsEnabled; help = $c.HelpText; class = $c.ClassName; automationId = $c.AutomationId })
    } catch { }
}
$rows | ConvertTo-Json -Depth 3 -Compress | Out-File -FilePath $Out -Encoding utf8
