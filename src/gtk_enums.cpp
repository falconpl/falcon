/**
 *  \file gtk_enums.cpp
 */

#include "gtk_enums.hpp"

#include <gtk/gtk.h>

namespace Falcon {
namespace Gtk {

void Enums::modInit( Falcon::Module* mod )
{

    Gtk::ConstIntTab intConstants[] =
    {

    /*
     *  GtkAccelFlags
     */
    { "ACCEL_VISIBLE",      GTK_ACCEL_VISIBLE },
    { "ACCEL_LOCKED",       GTK_ACCEL_LOCKED },
    { "ACCEL_MASK",         GTK_ACCEL_MASK },

    /*
     *  GtkAnchorType
     */
    { "ANCHOR_CENTER",      GTK_ANCHOR_CENTER },
    { "ANCHOR_NORTH",       GTK_ANCHOR_NORTH },
    { "ANCHOR_NORTH_WEST",  GTK_ANCHOR_NORTH_WEST },
    { "ANCHOR_NORTH_EAST",  GTK_ANCHOR_NORTH_EAST },
    { "ANCHOR_SOUTH",       GTK_ANCHOR_SOUTH },
    { "ANCHOR_SOUTH_WEST",  GTK_ANCHOR_SOUTH_WEST },
    { "ANCHOR_SOUTH_EAST",  GTK_ANCHOR_SOUTH_EAST },
    { "ANCHOR_WEST",        GTK_ANCHOR_WEST },
    { "ANCHOR_EAST",        GTK_ANCHOR_EAST },
    { "ANCHOR_N",           GTK_ANCHOR_N },
    { "ANCHOR_NW",          GTK_ANCHOR_NW },
    { "ANCHOR_NE",          GTK_ANCHOR_NE },
    { "ANCHOR_S",           GTK_ANCHOR_S },
    { "ANCHOR_SW",          GTK_ANCHOR_SW },
    { "ANCHOR_SE",          GTK_ANCHOR_SE },
    { "ANCHOR_W",           GTK_ANCHOR_W },
    { "ANCHOR_E",           GTK_ANCHOR_E },

#if GTK_MINOR_VERSION >= 16
    /*
     *  GtkArrowPlacement
     */
    { "ARROWS_BOTH",        GTK_ARROWS_BOTH },
    { "ARROWS_START",       GTK_ARROWS_START },
    { "ARROWS_END",         GTK_ARROWS_END },
#endif

    /*
     *  GtkArrowType
     */
    { "ARROW_UP",           GTK_ARROW_UP },
    { "ARROW_DOWN",         GTK_ARROW_DOWN },
    { "ARROW_LEFT",         GTK_ARROW_LEFT },
    { "ARROW_RIGHT",        GTK_ARROW_RIGHT },
    { "ARROW_NONE",         GTK_ARROW_NONE },

    /*
     *  GtkAttachOptions
     */
    { "EXPAND",             GTK_EXPAND },
    { "SHRINK",             GTK_SHRINK },
    { "FILL",               GTK_FILL },

    /*
     *  GtkButtonBoxStyle
     */
    { "BUTTONBOX_DEFAULT_STYLE",    GTK_BUTTONBOX_DEFAULT_STYLE },
    { "BUTTONBOX_SPREAD",           GTK_BUTTONBOX_SPREAD },
    { "BUTTONBOX_EDGE",             GTK_BUTTONBOX_EDGE },
    { "BUTTONBOX_START",            GTK_BUTTONBOX_START },
    { "BUTTONBOX_END",              GTK_BUTTONBOX_END },
    { "BUTTONBOX_CENTER",           GTK_BUTTONBOX_CENTER },

    /*
     *  GtkCornerType
     */
    { "CORNER_TOP_LEFT",        GTK_CORNER_TOP_LEFT },
    { "CORNER_BOTTOM_LEFT",     GTK_CORNER_BOTTOM_LEFT },
    { "CORNER_TOP_RIGHT",       GTK_CORNER_TOP_RIGHT },
    { "CORNER_BOTTOM_RIGHT",    GTK_CORNER_BOTTOM_RIGHT },

    /*
     *  GtkCurveType
     */
    { "CURVE_TYPE_LINEAR",      GTK_CURVE_TYPE_LINEAR },
    { "CURVE_TYPE_SPLINE",      GTK_CURVE_TYPE_SPLINE },
    { "CURVE_TYPE_FREE",        GTK_CURVE_TYPE_FREE },

    /*
     *  GtkDeleteType
     */
    { "DELETE_CHARS",           GTK_DELETE_CHARS },
    { "DELETE_WORD_ENDS",       GTK_DELETE_WORD_ENDS },
    { "DELETE_WORDS",           GTK_DELETE_WORDS },
    { "DELETE_DISPLAY_LINES",   GTK_DELETE_DISPLAY_LINES },
    { "DELETE_DISPLAY_LINE_ENDS",GTK_DELETE_DISPLAY_LINE_ENDS },
    { "DELETE_PARAGRAPH_ENDS",  GTK_DELETE_PARAGRAPH_ENDS },
    { "DELETE_PARAGRAPHS",      GTK_DELETE_PARAGRAPHS },
    { "DELETE_WHITESPACE",      GTK_DELETE_WHITESPACE },

    /*
     *  GtkDirectionType
     */
    { "DIR_TAB_FORWARD",        GTK_DIR_TAB_FORWARD },
    { "DIR_TAB_BACKWARD",       GTK_DIR_TAB_BACKWARD },
    { "DIR_UP",                 GTK_DIR_UP },
    { "DIR_DOWN",               GTK_DIR_DOWN },
    { "DIR_LEFT",               GTK_DIR_LEFT },
    { "DIR_RIGHT",              GTK_DIR_RIGHT },

    /*
     *  GtkExpanderStyle
     */
    { "EXPANDER_COLLAPSED",     GTK_EXPANDER_COLLAPSED },
    { "EXPANDER_SEMI_COLLAPSED",GTK_EXPANDER_SEMI_COLLAPSED },
    { "EXPANDER_SEMI_EXPANDED", GTK_EXPANDER_SEMI_EXPANDED },
    { "EXPANDER_EXPANDED",      GTK_EXPANDER_EXPANDED },

    /*
     *  GtkIMPreeditStyle
     */
    { "IM_PREEDIT_NOTHING",     GTK_IM_PREEDIT_NOTHING },
    { "IM_PREEDIT_CALLBACK",    GTK_IM_PREEDIT_CALLBACK },
    { "IM_PREEDIT_NONE",        GTK_IM_PREEDIT_NONE },

    /*
     *  GtkIMStatusStyle
     */
    { "IM_STATUS_NOTHING",      GTK_IM_STATUS_NOTHING },
    { "IM_STATUS_CALLBACK",     GTK_IM_STATUS_CALLBACK },
    { "IM_STATUS_NONE",         GTK_IM_STATUS_NONE },

    /*
     *  GtkJustification
     */
    { "JUSTIFY_LEFT",       GTK_JUSTIFY_LEFT },
    { "JUSTIFY_RIGHT",      GTK_JUSTIFY_RIGHT },
    { "JUSTIFY_CENTER",     GTK_JUSTIFY_CENTER },
    { "JUSTIFY_FILL",       GTK_JUSTIFY_FILL },

    /*
     *  GtkMatchType
     */
    { "MATCH_ALL",          GTK_MATCH_ALL },
    { "MATCH_ALL_TAIL",     GTK_MATCH_ALL_TAIL },
    { "MATCH_HEAD",         GTK_MATCH_HEAD },
    { "MATCH_TAIL",         GTK_MATCH_TAIL },
    { "MATCH_EXACT",        GTK_MATCH_EXACT },
    { "MATCH_LAST",         GTK_MATCH_LAST },

    /*
     *  GtkMetricType
     */
    { "PIXELS",     GTK_PIXELS },
    { "INCHES",     GTK_INCHES },
    { "CENTIMETERS",GTK_CENTIMETERS },

    /*
     *  GtkMovementStep
     */
    { "MOVEMENT_LOGICAL_POSITIONS",     GTK_MOVEMENT_LOGICAL_POSITIONS },
    { "MOVEMENT_VISUAL_POSITIONS",      GTK_MOVEMENT_VISUAL_POSITIONS },
    { "MOVEMENT_WORDS",                 GTK_MOVEMENT_WORDS },
    { "MOVEMENT_DISPLAY_LINES",         GTK_MOVEMENT_DISPLAY_LINES },
    { "MOVEMENT_DISPLAY_LINE_ENDS",     GTK_MOVEMENT_DISPLAY_LINE_ENDS },
    { "MOVEMENT_PARAGRAPHS",            GTK_MOVEMENT_PARAGRAPHS },
    { "MOVEMENT_PARAGRAPH_ENDS",        GTK_MOVEMENT_PARAGRAPH_ENDS },
    { "MOVEMENT_PAGES",                 GTK_MOVEMENT_PAGES },
    { "MOVEMENT_BUFFER_ENDS",           GTK_MOVEMENT_BUFFER_ENDS },
    { "MOVEMENT_HORIZONTAL_PAGES",      GTK_MOVEMENT_HORIZONTAL_PAGES },

    /*
     *  GtkOrientation
     */
    { "ORIENTATION_HORIZONTAL", GTK_ORIENTATION_HORIZONTAL },
    { "ORIENTATION_VERTICAL",   GTK_ORIENTATION_VERTICAL },

    /*
     *  GtkPackType
     */
    { "PACK_START",     GTK_PACK_START },
    { "PACK_END",       GTK_PACK_END },

    /*
     *  GtkPathPriorityType
     */
    { "PATH_PRIO_LOWEST",       GTK_PATH_PRIO_LOWEST },
    { "PATH_PRIO_GTK",          GTK_PATH_PRIO_GTK },
    { "PATH_PRIO_APPLICATION",  GTK_PATH_PRIO_APPLICATION },
    { "PATH_PRIO_THEME",        GTK_PATH_PRIO_THEME },
    { "PATH_PRIO_RC",           GTK_PATH_PRIO_RC },
    { "PATH_PRIO_HIGHEST",      GTK_PATH_PRIO_HIGHEST },

    /*
     *  GtkPathType
     */
    { "PATH_WIDGET",        GTK_PATH_WIDGET },
    { "PATH_WIDGET_CLASS",  GTK_PATH_WIDGET_CLASS },
    { "PATH_CLASS",         GTK_PATH_CLASS },

    /*
     *  GtkPolicyType
     */
    { "POLICY_ALWAYS",      GTK_POLICY_ALWAYS },
    { "POLICY_AUTOMATIC",   GTK_POLICY_AUTOMATIC },
    { "POLICY_NEVER",       GTK_POLICY_NEVER },

    /*
     *  GtkPositionType
     */
    { "POS_LEFT",       GTK_POS_LEFT },
    { "POS_RIGHT",      GTK_POS_RIGHT },
    { "POS_TOP",        GTK_POS_TOP },
    { "POS_BOTTOM",     GTK_POS_BOTTOM },

    /*
     *  GtkPreviewType
     */
    { "PREVIEW_COLOR",      GTK_PREVIEW_COLOR },
    { "PREVIEW_GRAYSCALE",  GTK_PREVIEW_GRAYSCALE },

    /*
     *  GtkReliefStyle
     */
    { "RELIEF_NORMAL",      GTK_RELIEF_NORMAL },
    { "RELIEF_HALF",        GTK_RELIEF_HALF },
    { "RELIEF_NONE",        GTK_RELIEF_NONE },

    /*
     *  GtkResizeMode
     */
    { "RESIZE_PARENT",      GTK_RESIZE_PARENT },
    { "RESIZE_QUEUE",       GTK_RESIZE_QUEUE },
    { "RESIZE_IMMEDIATE",   GTK_RESIZE_IMMEDIATE },

    /*
     *  GtkScrollStep
     */
    { "SCROLL_STEPS",       GTK_SCROLL_STEPS },
    { "SCROLL_PAGES",       GTK_SCROLL_PAGES },
    { "SCROLL_ENDS",        GTK_SCROLL_ENDS },
    { "SCROLL_HORIZONTAL_STEPS",    GTK_SCROLL_HORIZONTAL_STEPS },
    { "SCROLL_HORIZONTAL_PAGES",    GTK_SCROLL_HORIZONTAL_PAGES },
    { "SCROLL_HORIZONTAL_ENDS",     GTK_SCROLL_HORIZONTAL_ENDS },

    /*
     *  GtkScrollType
     */
    { "SCROLL_NONE",            GTK_SCROLL_NONE },
    { "SCROLL_JUMP",            GTK_SCROLL_JUMP },
    { "SCROLL_STEP_BACKWARD",   GTK_SCROLL_STEP_BACKWARD },
    { "SCROLL_STEP_FORWARD",    GTK_SCROLL_STEP_FORWARD },
    { "SCROLL_PAGE_BACKWARD",   GTK_SCROLL_PAGE_BACKWARD },
    { "SCROLL_PAGE_FORWARD",    GTK_SCROLL_PAGE_FORWARD },
    { "SCROLL_STEP_UP",         GTK_SCROLL_STEP_UP },
    { "SCROLL_STEP_DOWN",       GTK_SCROLL_STEP_DOWN },
    { "SCROLL_PAGE_UP",         GTK_SCROLL_PAGE_UP },
    { "SCROLL_PAGE_DOWN",       GTK_SCROLL_PAGE_DOWN },
    { "SCROLL_STEP_LEFT",       GTK_SCROLL_STEP_LEFT },
    { "SCROLL_STEP_RIGHT",      GTK_SCROLL_STEP_RIGHT },
    { "SCROLL_PAGE_LEFT",       GTK_SCROLL_PAGE_LEFT },
    { "SCROLL_PAGE_RIGHT",      GTK_SCROLL_PAGE_RIGHT },
    { "SCROLL_START",           GTK_SCROLL_START },
    { "SCROLL_END",             GTK_SCROLL_END },

    /*
     *  GtkSelectionMode
     */
    { "SELECTION_NONE",         GTK_SELECTION_NONE },
    { "SELECTION_SINGLE",       GTK_SELECTION_SINGLE },
    { "SELECTION_BROWSE",       GTK_SELECTION_BROWSE },
    { "SELECTION_MULTIPLE",     GTK_SELECTION_MULTIPLE },
    { "SELECTION_EXTENDED",     GTK_SELECTION_EXTENDED },

    /*
     *  GtkShadowType
     */
    { "SHADOW_NONE",        GTK_SHADOW_NONE },
    { "SHADOW_IN",          GTK_SHADOW_IN },
    { "SHADOW_OUT",         GTK_SHADOW_OUT },
    { "SHADOW_ETCHED_IN",   GTK_SHADOW_ETCHED_IN },
    { "SHADOW_ETCHED_OUT",  GTK_SHADOW_ETCHED_OUT },

    /*
     *  GtkSideType
     */
    { "SIDE_TOP",       GTK_SIDE_TOP },
    { "SIDE_BOTTOM",    GTK_SIDE_BOTTOM },
    { "SIDE_LEFT",      GTK_SIDE_LEFT },
    { "SIDE_RIGHT",     GTK_SIDE_RIGHT },

    /*
     *  GtkStateType
     */
    { "STATE_NORMAL",       GTK_STATE_NORMAL },
    { "STATE_ACTIVE",       GTK_STATE_ACTIVE },
    { "STATE_PRELIGHT",     GTK_STATE_PRELIGHT },
    { "STATE_SELECTED",     GTK_STATE_SELECTED },
    { "STATE_INSENSITIVE",  GTK_STATE_INSENSITIVE },

    /*
     *  GtkSubmenuDirection
     */
    { "DIRECTION_LEFT",     GTK_DIRECTION_LEFT },
    { "DIRECTION_RIGHT",    GTK_DIRECTION_RIGHT },

    /*
     *  GtkSubmenuPlacement
     */
    { "TOP_BOTTOM",     GTK_TOP_BOTTOM },
    { "LEFT_RIGHT",     GTK_LEFT_RIGHT },

    /*
     *  GtkToolbarStyle
     */
    { "TOOLBAR_ICONS",      GTK_TOOLBAR_ICONS },
    { "TOOLBAR_TEXT",       GTK_TOOLBAR_TEXT },
    { "TOOLBAR_BOTH",       GTK_TOOLBAR_BOTH },
    { "TOOLBAR_BOTH_HORIZ", GTK_TOOLBAR_BOTH_HORIZ },

    /*
     *  GtkUpdateType
     */
    { "UPDATE_CONTINUOUS",      GTK_UPDATE_CONTINUOUS },
    { "UPDATE_DISCONTINUOUS",   GTK_UPDATE_DISCONTINUOUS },
    { "UPDATE_DELAYED",         GTK_UPDATE_DELAYED },

    /*
     *  GtkVisibility
     */
    { "VISIBILITY_NONE",    GTK_VISIBILITY_NONE },
    { "VISIBILITY_PARTIAL", GTK_VISIBILITY_PARTIAL },
    { "VISIBILITY_FULL",    GTK_VISIBILITY_FULL },

    /*
     *  GtkWindowPosition
     */
    { "WIN_POS_NONE",       GTK_WIN_POS_NONE },
    { "WIN_POS_CENTER",     GTK_WIN_POS_CENTER },
    { "WIN_POS_MOUSE",      GTK_WIN_POS_MOUSE },
    { "WIN_POS_CENTER_ALWAYS",  GTK_WIN_POS_CENTER_ALWAYS },
    { "WIN_POS_CENTER_ON_PARENT",   GTK_WIN_POS_CENTER_ON_PARENT },

    /*
     *  GtkWindowType
     */
    { "WINDOW_TOPLEVEL",    GTK_WINDOW_TOPLEVEL },
    { "WINDOW_POPUP",       GTK_WINDOW_POPUP },

    /*
     *  GtkSortType
     */
    { "SORT_ASCENDING",     GTK_SORT_ASCENDING },
    { "SORT_DESCENDING",    GTK_SORT_DESCENDING },

    /*
     *  GtkDragResult
     */
    { "DRAG_RESULT_SUCCESS",        GTK_DRAG_RESULT_SUCCESS },
    { "DRAG_RESULT_NO_TARGET",      GTK_DRAG_RESULT_NO_TARGET },
    { "DRAG_RESULT_USER_CANCELLED", GTK_DRAG_RESULT_USER_CANCELLED },
    { "DRAG_RESULT_TIMEOUT_EXPIRED",GTK_DRAG_RESULT_TIMEOUT_EXPIRED },
    { "DRAG_RESULT_GRAB_BROKEN",    GTK_DRAG_RESULT_GRAB_BROKEN },
    { "DRAG_RESULT_ERROR",          GTK_DRAG_RESULT_ERROR },

    /*
     *  GtkWidgetFlags
     */
    { "TOPLEVEL",           GTK_TOPLEVEL },
    { "NO_WINDOW",          GTK_NO_WINDOW },
    { "REALIZED",           GTK_REALIZED },
    { "MAPPED",             GTK_MAPPED },
    { "VISIBLE",            GTK_VISIBLE },
    { "SENSITIVE",          GTK_SENSITIVE },
    { "PARENT_SENSITIVE",   GTK_PARENT_SENSITIVE },
    { "CAN_FOCUS",          GTK_CAN_FOCUS },
    { "HAS_FOCUS",          GTK_HAS_FOCUS },
    { "CAN_DEFAULT",        GTK_CAN_DEFAULT },
    { "HAS_DEFAULT",        GTK_HAS_DEFAULT },
    { "HAS_GRAB",           GTK_HAS_GRAB },
    { "RC_STYLE",           GTK_RC_STYLE },
    { "COMPOSITE_CHILD",    GTK_COMPOSITE_CHILD },
    { "NO_REPARENT",        GTK_NO_REPARENT },
    { "APP_PAINTABLE",      GTK_APP_PAINTABLE },
    { "RECEIVES_DEFAULT",   GTK_RECEIVES_DEFAULT },
    { "DOUBLE_BUFFERED",    GTK_DOUBLE_BUFFERED },
    { "NO_SHOW_ALL",        GTK_NO_SHOW_ALL },

    /*
     *  GtkTextDirection
     */
    { "TEXT_DIR_NONE",      GTK_TEXT_DIR_NONE },
    { "TEXT_DIR_LTR",       GTK_TEXT_DIR_LTR },
    { "TEXT_DIR_RTL",       GTK_TEXT_DIR_RTL },



    /*
     *  GdkEventType
     */
    { "GDK_NOTHING",        GDK_NOTHING },
    { "GDK_DELETE",         GDK_DELETE },
    { "GDK_DESTROY",        GDK_DESTROY },
    { "GDK_EXPOSE",         GDK_EXPOSE },
    { "GDK_MOTION_NOTIFY",  GDK_MOTION_NOTIFY },
    { "GDK_BUTTON_PRESS",   GDK_BUTTON_PRESS },
    { "GDK_2BUTTON_PRESS",  GDK_2BUTTON_PRESS },
    { "GDK_3BUTTON_PRESS",  GDK_3BUTTON_PRESS },
    { "GDK_BUTTON_RELEASE", GDK_BUTTON_RELEASE },
    { "GDK_KEY_PRESS",      GDK_KEY_PRESS },
    { "GDK_KEY_RELEASE",    GDK_KEY_RELEASE },
    { "GDK_ENTER_NOTIFY",   GDK_ENTER_NOTIFY },
    { "GDK_LEAVE_NOTIFY",   GDK_LEAVE_NOTIFY },
    { "GDK_FOCUS_CHANGE",   GDK_FOCUS_CHANGE },
    { "GDK_CONFIGURE",      GDK_CONFIGURE },
    { "GDK_MAP",            GDK_MAP },
    { "GDK_UNMAP",          GDK_UNMAP },
    { "GDK_PROPERTY_NOTIFY",GDK_PROPERTY_NOTIFY },
    { "GDK_SELECTION_CLEAR",GDK_SELECTION_CLEAR },
    { "GDK_SELECTION_REQUEST",GDK_SELECTION_REQUEST },
    { "GDK_SELECTION_NOTIFY",GDK_SELECTION_NOTIFY },
    { "GDK_PROXIMITY_IN",   GDK_PROXIMITY_IN },
    { "GDK_PROXIMITY_OUT",  GDK_PROXIMITY_OUT },
    { "GDK_DRAG_ENTER",     GDK_DRAG_ENTER },
    { "GDK_DRAG_LEAVE",     GDK_DRAG_LEAVE },
    { "GDK_DRAG_MOTION",    GDK_DRAG_MOTION },
    { "GDK_DRAG_STATUS",    GDK_DRAG_STATUS },
    { "GDK_DROP_START",     GDK_DROP_START },
    { "GDK_DROP_FINISHED",  GDK_DROP_FINISHED },
    { "GDK_CLIENT_EVENT",   GDK_CLIENT_EVENT },
    { "GDK_VISIBILITY_NOTIFY",GDK_VISIBILITY_NOTIFY },
    { "GDK_NO_EXPOSE",      GDK_NO_EXPOSE },
    { "GDK_SCROLL",         GDK_SCROLL },
    { "GDK_WINDOW_STATE",   GDK_WINDOW_STATE },
    { "GDK_SETTING",        GDK_SETTING },
    #if GTK_MINOR_VERSION >= 6
    { "GDK_OWNER_CHANGE",   GDK_OWNER_CHANGE },
    #endif
    #if GTK_MINOR_VERSION >= 8
    { "GDK_GRAB_BROKEN",    GDK_GRAB_BROKEN },
    #endif
    #if GTK_MINOR_VERSION >= 14
    { "GDK_DAMAGE",         GDK_DAMAGE },
    #endif
    #if GTK_MINOR_VERSION >= 18
    { "GDK_EVENT_LAST",     GDK_EVENT_LAST },
    #endif

    /*
     *  GdkGravity
     */
    { "GDK_GRAVITY_NORTH_WEST", GDK_GRAVITY_NORTH_WEST },
    { "GDK_GRAVITY_NORTH",      GDK_GRAVITY_NORTH },
    { "GDK_GRAVITY_NORTH_EAST", GDK_GRAVITY_NORTH_EAST },
    { "GDK_GRAVITY_WEST",       GDK_GRAVITY_WEST },
    { "GDK_GRAVITY_CENTER",     GDK_GRAVITY_CENTER },
    { "GDK_GRAVITY_EAST",       GDK_GRAVITY_EAST },
    { "GDK_GRAVITY_SOUTH_WEST", GDK_GRAVITY_SOUTH_WEST },
    { "GDK_GRAVITY_SOUTH",      GDK_GRAVITY_SOUTH },
    { "GDK_GRAVITY_SOUTH_EAST", GDK_GRAVITY_SOUTH_EAST },
    { "GDK_GRAVITY_STATIC",     GDK_GRAVITY_STATIC },



    { NULL, 0 }
    };

    for ( Gtk::ConstIntTab* tab = intConstants; tab->name; ++tab )
        mod->addConstant( tab->name, tab->value );
}


} // Gtk
} // Falcon
