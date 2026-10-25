<#
.SYNOPSIS
    Drags one file over a window with a REAL operating-system drag (OLE), so the packaged app's drop overlay can be looked at. Used by packaged_visual_check.py.
.DESCRIPTION
    A small borderless window is placed at -FormX/-FormY, the left mouse button is pressed on it, the file is dragged to -TargetX/-TargetY (screen pixels over the app window), the
    cursor stays there until "<Flag>.done" exists (the caller takes its picture of the app's own window meanwhile), then the drag is cancelled with Esc: nothing is dropped.
    It writes "<Flag>.hover" when the cursor is over the target. The mouse moves for about three seconds; nothing is clicked anywhere else.
#>
param(
    [Parameter(Mandatory = $true)][string]$File,
    [Parameter(Mandatory = $true)][int]$FormX,
    [Parameter(Mandatory = $true)][int]$FormY,
    [Parameter(Mandatory = $true)][int]$TargetX,
    [Parameter(Mandatory = $true)][int]$TargetY,
    [Parameter(Mandatory = $true)][string]$Flag,
    [switch]$Drop
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -ReferencedAssemblies System.Windows.Forms, System.Drawing @"
using System;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using System.Drawing;
using System.Runtime.InteropServices;

public static class DragHelper
{
    [DllImport("user32.dll")] static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] static extern int GetSystemMetrics(int index);
    // a real (hardware-style) move: the drag loop of the operating system sees it, a bare SetCursorPos is not always noticed
    static void MoveTo(int x, int y)
    {
        int w = GetSystemMetrics(0), h = GetSystemMetrics(1);
        mouse_event(0x0001 | 0x8000, (uint)(x * 65535 / Math.Max(w - 1, 1)), (uint)(y * 65535 / Math.Max(h - 1, 1)), 0, UIntPtr.Zero);
    }
    [DllImport("user32.dll")] static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
    [DllImport("user32.dll")] static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);

    public static void Run(string file, int formX, int formY, int targetX, int targetY, string flag, bool drop)
    {
        var form = new Form();
        form.FormBorderStyle = FormBorderStyle.None;
        form.StartPosition = FormStartPosition.Manual;
        form.Location = new Point(formX, formY);
        form.Size = new Size(120, 40);
        form.TopMost = true;
        form.ShowInTaskbar = false;
        form.BackColor = Color.FromArgb(14, 138, 142);
        form.Show();
        Application.DoEvents();
        var choreography = new Thread(() =>
        {
            Thread.Sleep(700);                                               // the drag loop is running now
            MoveTo(formX + 70, formY + 24);                                  // the first small move starts the drag
            Thread.Sleep(250);
            MoveTo(targetX - 200, targetY - 40);
            Thread.Sleep(250);
            MoveTo(targetX - 60, targetY);
            Thread.Sleep(250);
            MoveTo(targetX - 10, targetY);
            Thread.Sleep(250);
            MoveTo(targetX, targetY);
            File.WriteAllText(flag + ".hover", "1");
            for (int i = 0; i < 200 && !File.Exists(flag + ".done"); i++) Thread.Sleep(50);
            if (!drop)
            {
                keybd_event(0x1B, 0, 0, UIntPtr.Zero);                       // Esc: cancel, nothing is dropped
                keybd_event(0x1B, 0, 2, UIntPtr.Zero);
            }
            Thread.Sleep(300);
            mouse_event(0x0004, 0, 0, 0, UIntPtr.Zero);                     // left button up
        });
        var data = new DataObject();
        var list = new System.Collections.Specialized.StringCollection();
        list.Add(file);
        data.SetFileDropList(list);
        SetCursorPos(formX + 60, formY + 20);
        Thread.Sleep(300);
        mouse_event(0x0002, 0, 0, 0, UIntPtr.Zero);                         // left button down on the helper window
        Thread.Sleep(150);
        choreography.Start();
        try
        {
            var effect = form.DoDragDrop(data, DragDropEffects.Copy);
            File.WriteAllText(flag + ".result", effect.ToString());
        }
        catch (Exception ex)
        {
            File.WriteAllText(flag + ".error", ex.ToString());
        }
        choreography.Join(5000);
        form.Close();
    }
}
"@

[DragHelper]::Run($File, $FormX, $FormY, $TargetX, $TargetY, $Flag, [bool]$Drop)
