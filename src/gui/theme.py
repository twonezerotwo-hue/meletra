"""Look of the app: the MELETRA design system (``tokens``) turned into one style sheet on top of Qt's Fusion style, plus the app icon.

Every colour, size, radius and spacing comes from ``tokens``; nothing here is invented per widget. Dark navy is the brand look, light is the same roles in light colours.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QImage, QPalette, QPixmap

from . import focus, icons, tokens
from .tokens import FONT_FAMILY, FONT_PX as F, HEIGHT as H, RADIUS as R

UI_CACHE = Path(tempfile.gettempdir()) / "MELETRA-ui"


def effective_theme(setting: str) -> str:
    """``light`` or ``dark`` for the setting ``system`` | ``light`` | ``dark``."""
    if setting in ("light", "dark"):
        return setting
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        return "dark" if scheme == Qt.ColorScheme.Dark else "light"
    except Exception:  # noqa: BLE001 - older Qt / headless
        return "light"


def _rgba(hex_color: str, alpha: int) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


def stylesheet(theme: str) -> str:
    c = tokens.PALETTES["dark" if theme == "dark" else "light"]
    chevron = icons.png_file("chevron", c.text2, 14, UI_CACHE)
    chevron_up = icons.png_file("chevron_up", c.text2, 14, UI_CACHE)
    check = icons.png_file("check", c.on_teal, 14, UI_CACHE)
    nav_selected = _rgba(c.teal, 30)
    return f"""
    QWidget {{ font-family: {FONT_FAMILY}; font-size: {F['body']}px; color: {c.text}; }}
    QMainWindow, QDialog, QStackedWidget, QScrollArea {{ background: {c.bg}; }}
    QMenuBar {{ background: {c.sidebar}; color: {c.text2}; border-bottom: 1px solid {c.border}; }}
    QMenuBar::item {{ padding: 6px 10px; background: transparent; }}
    QMenuBar::item:selected {{ background: {c.surface_hover}; color: {c.text}; border-radius: 6px; }}
    QStatusBar {{ background: {c.sidebar}; color: {c.text2}; border-top: 1px solid {c.border}; font-size: {F['meta']}px; }}
    QStatusBar::item {{ border: none; }}

    /* ---- type ---- */
    QLabel#hero {{ font-size: {F['page']}px; font-weight: 600; color: {c.text}; }}
    QLabel#title {{ font-size: {F['title']}px; font-weight: 600; }}
    QLabel#sectionTitle {{ font-size: {F['section']}px; font-weight: 600; }}
    QLabel#subtitle, QLabel#muted {{ color: {c.text2}; font-size: {F['secondary']}px; }}
    QLabel#meta {{ color: {c.text3}; font-size: {F['meta']}px; }}
    QLabel#bigNumber {{ font-size: {F['section']}px; font-weight: 600; }}
    QLabel#percentBig {{ font-size: {F['display']}px; font-weight: 600; color: {c.teal}; }}
    QLabel#percent {{ font-size: {F['section']}px; font-weight: 600; color: {c.teal}; min-width: 48px; }}
    QLabel#statusLine {{ font-size: {F['body']}px; color: {c.text2}; }}
    QLabel#fileName {{ font-size: {F['name']}px; font-weight: 600; }}
    QLabel#fileType {{ color: {c.text2}; font-size: {F['secondary']}px; }}
    QLabel#brandName {{ font-size: {F['name']}px; font-weight: 600; }}

    /* ---- surfaces ---- */
    QFrame#card {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['card']}px; }}
    QFrame#sideRail {{ background: {c.sidebar}; border-right: 1px solid {c.border}; }}
    QFrame#railDivider {{ background: {c.border}; max-height: 1px; min-height: 1px; border: none; }}
    QFrame#studyHeader {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['card']}px; }}
    QFrame#statCard {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['card']}px; }}
    QFrame#studyRail {{ background: transparent; border: none; min-width: 220px; }}
    QFrame#railPanel {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['card']}px; }}
    QTextBrowser#reading {{ background: transparent; border: none; padding: 0; }}
    QTextBrowser#reading:focus {{ border: none; }}
    QFrame#flashCard {{ background: {c.surface}; border: 1px solid {c.border_strong}; border-radius: {R['drop']}px; }}
    QFrame#fileCard {{ background: {c.surface}; border: 1px solid {c.border_strong}; border-radius: {R['card']}px; }}
    QFrame#queueRow {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['control']}px; }}
    QFrame#queueRow[current="true"] {{ border: 1px solid {c.teal}; }}
    QFrame#emptyState {{ background: transparent; border: 1px dashed {c.border_strong}; border-radius: {R['card']}px; }}
    QLabel#studyTitle {{ font-size: {F['title']}px; font-weight: 600; }}
    QLabel#studyStat {{ font-size: {F['lead']}px; font-weight: 600; color: {c.teal}; }}
    QLabel#flashFront {{ font-size: {F['title']}px; font-weight: 600; padding: 8px 16px; }}
    QLabel#flashBack {{ font-size: {F['section']}px; padding: 8px 16px; color: {c.text}; }}
    QLabel#flashCaption {{ color: {c.text3}; font-size: {F['meta']}px; font-weight: 500; }}
    QLabel#flashCaptionAnswer {{ color: {c.teal}; font-size: {F['meta']}px; font-weight: 600; }}
    QFrame#readingDivider {{ background: {c.border}; max-height: 1px; min-height: 1px; border: none; }}
    QLabel#questionNumber {{ background: {c.teal_soft}; color: {c.teal}; border-radius: 14px; min-width: 28px; max-width: 28px; min-height: 28px; max-height: 28px; font-size: {F['secondary']}px; font-weight: 600; }}
    QLabel#questionText {{ font-size: {F['section']}px; font-weight: 600; }}
    QLabel#focusQuestion {{ font-size: {F['lead']}px; font-weight: 600; }}
    QSplitter::handle {{ background: transparent; }}
    QListWidget#kindList {{ background: transparent; border: none; padding: 0; outline: 0; }}
    QListWidget#kindList::item {{ padding: 9px 12px; border-radius: {R['control']}px; color: {c.text2}; margin-bottom: 2px; }}
    QListWidget#kindList::item:hover {{ background: {c.surface}; color: {c.text}; }}
    QListWidget#kindList::item:selected {{ background: {_rgba(c.teal, 30)}; color: {c.text}; }}
    QListWidget#kindList::item:disabled {{ color: {c.text3}; }}
    QLabel#resultGood {{ color: {c.success}; font-weight: 600; font-size: {F['reading']}px; }}
    QLabel#resultBad {{ color: {c.error}; font-weight: 600; font-size: {F['reading']}px; }}

    /* ---- status chips and badges ---- */
    QLabel#statusChip {{ background: {c.neutral_bg}; color: {c.neutral}; border-radius: 11px; padding: 2px 10px; font-size: {F['meta']}px; font-weight: 600; min-height: 18px; }}
    QLabel#statusChip[state="completed"] {{ background: {c.success_bg}; color: {c.success}; }}
    QLabel#statusChip[state="running"], QLabel#statusChip[state="processing"] {{ background: {c.teal_soft}; color: {c.teal}; }}
    QLabel#statusChip[state="queued"] {{ background: {c.neutral_bg}; color: {c.neutral}; }}
    QLabel#statusChip[state="paused"] {{ background: {c.warning_bg}; color: {c.warning}; }}
    QLabel#statusChip[state="failed"] {{ background: {c.error_bg}; color: {c.error}; }}
    QLabel#badge {{ border-radius: 9px; padding: 1px 9px; font-size: {F['meta']}px; font-weight: 600; background: {c.neutral_bg}; color: {c.neutral}; }}
    QLabel#badge[kind="stable"] {{ background: {c.teal_soft}; color: {c.teal}; }}
    QLabel#badge[kind="experimental"] {{ background: {c.warning_bg}; color: {c.warning}; }}
    QLabel#badge[kind="on"] {{ background: {c.success_bg}; color: {c.success}; }}
    QLabel#badge[kind="off"] {{ background: {c.neutral_bg}; color: {c.neutral}; }}
    QLabel#stateBig {{ font-size: {F['lead']}px; font-weight: 600; }}
    QLabel#stateBig[kind="on"] {{ color: {c.success}; }}
    QLabel#stateBig[kind="off"] {{ color: {c.text2}; }}
    QLabel#banner, QLabel#bannerWarn, QLabel#bannerBad {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['control']}px; padding: 10px 12px; }}
    QLabel#banner {{ border-left: 4px solid {c.teal}; }}
    QLabel#bannerWarn {{ border-left: 4px solid {c.warning}; }}
    QLabel#bannerBad {{ border-left: 4px solid {c.error}; }}
    QLabel#toast {{ background: {c.surface_hover}; color: {c.text}; border: 1px solid {c.border_strong}; border-left: 4px solid {c.error}; border-radius: {R['control']}px; padding: 10px 16px; }}
    QLabel#badgeLocal {{ background: {c.teal_soft}; color: {c.teal}; border-radius: 10px; padding: 3px 12px; font-weight: 600; }}
    QLabel#badgeCloud {{ background: {c.warning_bg}; color: {c.warning}; border-radius: 10px; padding: 3px 12px; font-weight: 600; }}

    /* ---- buttons ---- */
    QPushButton {{ background: {c.surface}; border: 1px solid {c.border_strong}; border-radius: {R['control']}px; padding: 7px 16px; min-height: 22px; font-weight: 500; }}
    QPushButton:hover {{ background: {c.surface_hover}; border-color: {c.teal}; }}
    QPushButton:pressed {{ background: {c.field}; }}
    QPushButton:disabled {{ color: {c.text3}; background: {c.surface}; border-color: {c.border}; }}
    QPushButton#primary {{ background: {c.teal}; color: {c.on_teal}; border: 1px solid {c.teal}; font-weight: 600; padding: 7px 22px; }}
    QPushButton#primary:hover {{ background: {c.teal_hover}; border-color: {c.teal_hover}; }}
    QPushButton#primary:pressed {{ background: {c.teal_press}; }}
    QPushButton#primary:disabled {{ background: {c.surface_hover}; color: {c.text3}; border-color: {c.border}; }}
    QPushButton#big {{ background: {c.teal}; color: {c.on_teal}; border: 1px solid {c.teal}; border-radius: {R['card']}px; font-size: {F['cta']}px; font-weight: 600;
                      padding: 14px 30px; min-height: 22px; }}
    QPushButton#big:hover {{ background: {c.teal_hover}; border-color: {c.teal_hover}; }}
    QPushButton#big:pressed {{ background: {c.teal_press}; }}
    QPushButton#big:disabled {{ background: {c.surface_hover}; color: {c.text3}; border-color: {c.border}; }}
    QPushButton#danger {{ background: {c.error_bg}; color: {c.error}; border: 1px solid {c.error}; font-weight: 600; padding: 7px 22px; }}
    QPushButton#danger:hover {{ background: {c.error}; color: {c.on_teal}; }}
    QPushButton#link {{ background: transparent; border: 1px solid transparent; color: {c.teal}; padding: 4px 8px; min-height: 18px; font-weight: 500; }}
    QPushButton#link:hover {{ background: transparent; color: {c.teal_hover}; border-color: transparent; }}
    QPushButton#link:disabled {{ color: {c.text3}; }}
    QPushButton#iconButton {{ background: transparent; border: 1px solid transparent; border-radius: {R['small']}px; padding: 4px; min-height: 20px; min-width: 20px; }}
    QPushButton#iconButton:hover {{ background: {c.surface_hover}; border-color: {c.border_strong}; }}
    QPushButton#chip {{ background: {c.teal_soft}; border: 1px solid transparent; border-radius: 11px; padding: 1px 10px; min-height: 18px; font-size: {F['meta']}px; color: {c.teal}; }}
    QPushButton#chip:hover {{ background: {c.surface_hover}; border-color: {c.teal}; }}
    QPushButton#tile {{ text-align: left; padding: 0; min-height: 72px; border-radius: {R['card']}px; background: {c.surface}; border: 1px solid {c.border}; }}
    QPushButton#tile:hover {{ background: {c.surface_hover}; border-color: {c.teal}; }}
    QPushButton#tile:disabled {{ background: {c.surface}; border-color: {c.border}; }}
    QPushButton#tile:focus {{ border: 2px solid {c.gold}; padding: 0; }}
    QLabel#tileTitle {{ font-size: {F['body']}px; font-weight: 600; background: transparent; }}
    QLabel#tileSub {{ font-size: {F['meta']}px; color: {c.text2}; background: transparent; }}
    QPushButton#option {{ text-align: left; padding: 12px 16px; border-radius: {R['card']}px; min-height: 22px; }}
    QPushButton#privacyChip {{ background: {c.surface}; color: {c.text2}; border: 1px solid {c.border}; border-radius: 15px; padding: 4px 14px 4px 10px; min-height: 20px; font-size: {F['meta']}px; font-weight: 500; }}
    QPushButton#privacyChip:hover {{ border-color: {c.teal}; color: {c.text}; background: {c.surface}; }}
    QPushButton#navButton {{ background: transparent; border: none; border-left: 3px solid transparent; border-radius: {R['control']}px; text-align: left;
                            padding: 9px 12px; min-height: 22px; color: {c.text2}; font-weight: 500; }}
    QPushButton#navButton:hover {{ background: {c.surface}; color: {c.text}; border-left: 3px solid transparent; }}
    QPushButton#navButton:checked {{ background: {nav_selected}; color: {c.text}; font-weight: 600; border-left: 3px solid {c.teal}; }}
    QToolButton#menuButton {{ background: transparent; border: 1px solid transparent; border-radius: {R['small']}px; padding: 3px; }}
    QToolButton#menuButton:hover {{ background: {c.surface_hover}; border-color: {c.border_strong}; }}
    QToolButton#menuButton::menu-indicator {{ image: none; width: 0; }}

    /* ---- start screen ---- */
    QFrame#dropArea {{ background: transparent; border: none; }}
    QLabel#dropTitle {{ font-size: {F['lead']}px; font-weight: 600; background: transparent; }}
    QLabel#dropSub {{ color: {c.text2}; font-size: {F['secondary']}px; background: transparent; }}
    QLabel#fieldLabel {{ color: {c.text2}; font-size: {F['meta']}px; font-weight: 500; }}
    QPushButton#smallButton {{ padding: 5px 12px; min-height: 20px; font-size: {F['secondary']}px; }}
    QPushButton#dropBrowse {{ background: transparent; border: 1px solid {c.border_strong}; padding: 6px 18px; min-height: 20px; font-size: {F['secondary']}px; color: {c.text}; }}
    QPushButton#dropBrowse:hover {{ border-color: {c.teal}; color: {c.teal}; background: transparent; }}
    QLabel#toast[kind="info"] {{ border-left: 4px solid {c.teal}; }}

    /* ---- segmented control ---- */
    QFrame#segmented {{ background: {c.field}; border: 1px solid {c.border}; border-radius: {R['control']}px; }}
    QPushButton#segment {{ background: transparent; border: 1px solid transparent; border-radius: {R['small']}px; padding: 5px 14px; min-height: 20px; color: {c.text2}; font-weight: 500; }}
    QPushButton#segment:hover {{ background: {c.surface}; color: {c.text}; border-color: transparent; }}
    QPushButton#segment:checked {{ background: {c.teal_soft}; color: {c.teal}; border-color: transparent; }}

    /* ---- inputs ---- */
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {c.field}; border: 1px solid {c.border_strong}; border-radius: {R['control']}px; padding: 7px 12px;
                                                    min-height: 22px; selection-background-color: {c.teal}; selection-color: {c.on_teal}; }}
    QComboBox {{ combobox-popup: 0; }}                                    /* the list opens as a normal list under the box, so its items follow the style sheet (not the menu style's bright highlight) */
    QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {c.teal}; }}
    QComboBox:disabled, QLineEdit:disabled {{ color: {c.text3}; border-color: {c.border}; }}
    QSpinBox, QDoubleSpinBox {{ padding-right: 26px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 22px; border: none; background: transparent; }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 22px; border: none; background: transparent; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({chevron_up}); width: 12px; height: 12px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({chevron}); width: 12px; height: 12px; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 30px; border: none; }}
    QComboBox::down-arrow {{ image: url({chevron}); width: 14px; height: 14px; }}
    QComboBox QAbstractItemView {{ background: {c.surface}; border: 1px solid {c.border_strong}; border-radius: {R['control']}px; padding: 4px; outline: 0;
                                  selection-background-color: {c.surface_hover}; selection-color: {c.text}; }}
    QComboBox QAbstractItemView::item {{ min-height: 30px; padding: 0 10px; border-radius: {R['small']}px; }}
    QComboBox QAbstractItemView::item:selected {{ background: {c.surface_hover}; color: {c.text}; }}
    QCheckBox, QRadioButton {{ spacing: 10px; color: {c.text}; }}
    QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid {c.border_strong}; background: {c.field}; }}
    QCheckBox::indicator:hover {{ border-color: {c.teal}; }}
    QCheckBox::indicator:checked {{ background: {c.teal}; border-color: {c.teal}; image: url({check}); }}
    QRadioButton::indicator {{ width: 18px; height: 18px; border-radius: 10px; border: 1px solid {c.border_strong}; background: {c.field}; }}
    QRadioButton::indicator:hover {{ border-color: {c.teal}; }}
    QRadioButton::indicator:checked {{ border: 5px solid {c.teal}; background: {c.on_teal}; }}
    QProgressBar {{ background: {c.border}; border: none; border-radius: 5px; min-height: 10px; max-height: 10px; text-align: center; color: transparent; }}
    QProgressBar::chunk {{ background: {c.teal}; border-radius: 5px; }}

    /* ---- containers ---- */
    QTabWidget::pane {{ border: none; border-top: 1px solid {c.border}; background: transparent; top: -1px; }}
    QTabBar::tab {{ background: transparent; color: {c.text2}; padding: 10px 16px; margin-right: 4px; border: none; border-bottom: 2px solid transparent; font-weight: 500; }}
    QTabBar::tab:hover {{ color: {c.text}; }}
    QTabBar::tab:selected {{ color: {c.text}; border-bottom: 2px solid {c.teal}; font-weight: 600; }}
    QTabBar::tab:focus {{ border-bottom: 2px solid {c.focus}; }}
    QGroupBox {{ border: 1px solid {c.border}; border-radius: {R['card']}px; margin-top: 14px; padding: 16px 12px 12px 12px; background: {c.surface}; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; color: {c.text2}; font-weight: 600; }}
    QPlainTextEdit#log {{ background: {c.field}; color: {c.text2}; border: 1px solid {c.border}; border-radius: {R['control']}px; font-family: Consolas, Menlo, monospace;
                         font-size: {F['meta']}px; padding: 6px; }}
    QTextBrowser, QTextEdit, QPlainTextEdit {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['control']}px; padding: 10px; selection-background-color: {c.teal}; selection-color: {c.on_teal}; }}
    QPlainTextEdit#transcriptText {{ font-size: {F['reading']}px; padding: 18px 24px; border-radius: {R['card']}px; }}
    QListWidget#queueList {{ background: transparent; border: none; padding: 0; }}
    QListWidget#settingsNav {{ background: transparent; border: none; padding: 0; outline: 0; }}
    QListWidget#settingsNav::item {{ padding: 9px 12px; border-radius: {R['control']}px; color: {c.text2}; margin-bottom: 2px; font-weight: 500; }}
    QListWidget#settingsNav::item:hover {{ background: {c.surface}; color: {c.text}; }}
    QListWidget#settingsNav::item:selected {{ background: {_rgba(c.teal, 30)}; color: {c.text}; font-weight: 600; }}
    QListWidget {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['control']}px; padding: 4px; outline: 0; }}
    QListWidget::item {{ padding: 8px 10px; border-radius: {R['small']}px; }}
    QListWidget::item:selected {{ background: {c.surface_hover}; color: {c.text}; }}
    QTableWidget {{ background: {c.surface}; gridline-color: {c.border}; border: 1px solid {c.border}; border-radius: {R['control']}px; selection-background-color: {c.teal_soft}; selection-color: {c.text}; outline: 0; }}
    QTableWidget::item {{ padding: 6px 8px; }}
    QTableWidget::item:selected {{ background: {c.teal_soft}; color: {c.text}; }}
    QTableCornerButton::section {{ background: {c.bg}; border: none; }}
    QHeaderView::section {{ background: {c.bg}; border: none; border-bottom: 1px solid {c.border}; padding: 6px; color: {c.text2}; font-weight: 600; }}

    /* ---- menus, tooltips, scrollbars ---- */
    QMenu {{ background: {c.surface}; border: 1px solid {c.border_strong}; border-radius: {R['control']}px; padding: 6px; }}
    QMenu::item {{ padding: 8px 18px 8px 12px; border-radius: {R['small']}px; min-width: 150px; color: {c.text}; }}
    QMenu::item:selected {{ background: {c.surface_hover}; }}
    QMenu::item:disabled {{ color: {c.text3}; }}
    QMenu::icon {{ padding-left: 6px; }}
    QMenu::separator {{ height: 1px; background: {c.border}; margin: 6px 8px; }}
    QToolTip {{ background: {c.surface_hover}; color: {c.text}; border: 1px solid {c.border_strong}; padding: 6px 8px; font-size: {F['secondary']}px; }}
    QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {c.border_strong}; min-height: 36px; border-radius: 4px; margin: 0 2px; }}
    QScrollBar::handle:vertical:hover {{ background: {c.text3}; }}
    QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {c.border_strong}; min-width: 36px; border-radius: 4px; margin: 2px 0; }}
    QScrollBar::handle:horizontal:hover {{ background: {c.text3}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; background: none; border: none; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* ---- lecture cards ---- */
    QFrame#lectureCard {{ background: {c.surface}; border: 1px solid {c.border}; border-radius: {R['card']}px; }}
    QFrame#lectureCard:hover {{ background: {c.surface_hover}; border: 1px solid {_rgba(c.teal, 120)}; }}
    QLabel#lectureTitle {{ font-weight: 600; font-size: {F['body']}px; background: transparent; }}
    QLabel#lectureSub {{ color: {c.text2}; font-size: {F['meta']}px; background: transparent; }}
    QLabel#lectureMeta {{ color: {c.text3}; font-size: {F['meta']}px; background: transparent; }}
    QLineEdit#renameEdit {{ padding: 4px 8px; min-height: 20px; font-weight: 600; }}

    /* ---- keyboard focus: a teal 2 px ring, gold on the two teal-filled button kinds where teal would vanish (the padding shrinks by 1 px so nothing moves). Buttons only take the focus from the keyboard (src/gui/focus.py) ---- */
    QPushButton:focus {{ border: 2px solid {c.focus}; padding: 6px 15px; }}
    QPushButton#big:focus {{ border: 2px solid {c.gold}; padding: 13px 29px; }}
    QPushButton#primary:focus {{ border: 2px solid {c.gold}; padding: 6px 21px; }}
    QPushButton#link:focus {{ border: 2px solid {c.focus}; padding: 3px 7px; }}
    QPushButton#iconButton:focus, QToolButton:focus {{ border: 2px solid {c.focus}; padding: 2px; }}
    QPushButton#segment:focus {{ border: 2px solid {c.focus}; padding: 4px 13px; }}
    QPushButton#privacyChip:focus {{ border: 2px solid {c.focus}; padding: 3px 13px 3px 9px; }}
    QPushButton#smallButton:focus {{ border: 2px solid {c.focus}; padding: 4px 11px; }}
    QPushButton#dropBrowse:focus {{ border: 2px solid {c.focus}; padding: 5px 17px; }}
    QPushButton#navButton:focus {{ border: 2px solid {c.focus}; border-left: 3px solid {c.focus}; padding: 8px 11px; }}
    QFrame#lectureCard:focus {{ border: 2px solid {c.focus}; }}
    QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {c.focus}; padding: 6px 11px; }}
    QCheckBox:focus, QRadioButton:focus {{ color: {c.focus}; }}
    QListWidget:focus, QTextBrowser:focus, QPlainTextEdit:focus {{ border: 1px solid {c.focus}; }}
    """


def apply_theme(app, setting: str) -> str:
    """Set the palette and the style sheet for ``setting``. Applying a style sheet re-polishes every widget (about 0.8 s for the whole window), so nothing happens when
    the theme is already the one in use (Settings -> Save without a theme change)."""
    theme = effective_theme(setting)
    palette_tokens = tokens.set_current(theme)
    focus.install(app)                                  # Fusion with keyboard-only buttons
    sheet = stylesheet(theme)
    if app.styleSheet() == sheet and app.property("wm_theme") == theme:
        return theme
    p = palette_tokens
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(p.bg))
    palette.setColor(QPalette.WindowText, QColor(p.text))
    palette.setColor(QPalette.Base, QColor(p.field))
    palette.setColor(QPalette.AlternateBase, QColor(p.surface))
    palette.setColor(QPalette.Text, QColor(p.text))
    palette.setColor(QPalette.Button, QColor(p.surface))
    palette.setColor(QPalette.ButtonText, QColor(p.text))
    palette.setColor(QPalette.Highlight, QColor(p.teal))
    palette.setColor(QPalette.HighlightedText, QColor(p.on_teal))
    palette.setColor(QPalette.PlaceholderText, QColor(p.text3))
    palette.setColor(QPalette.Link, QColor(p.teal))
    palette.setColor(QPalette.ToolTipBase, QColor(p.surface_hover))
    palette.setColor(QPalette.ToolTipText, QColor(p.text))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(p.text3))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(p.text3))
    app.setPalette(palette)
    app.setStyleSheet(sheet)
    app.setProperty("wm_theme", theme)
    icons.refresh()
    return theme


def draw_icon(size: int = 256) -> QImage:
    """The MELETRA application icon (assets/brand/icons, made from the brand logo by tools/make_brand_assets.py) at ``size`` px."""
    from . import brand

    return brand.icon_image(size)


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(QPixmap.fromImage(draw_icon(size)))
    return icon
