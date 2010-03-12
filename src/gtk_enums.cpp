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
     *  GtkResizeMode
     */
    { "RESIZE_PARENT",      GTK_RESIZE_PARENT },
    { "RESIZE_QUEUE",       GTK_RESIZE_QUEUE },
    { "RESIZE_IMMEDIATE",   GTK_RESIZE_IMMEDIATE },

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
     *  GtkWindowType
     */
    { "WINDOW_TOPLEVEL",    GTK_WINDOW_TOPLEVEL },
    { "WINDOW_POPUP",       GTK_WINDOW_POPUP },

#if 0
    /*
     *  GdkEventType
     *  we might put Gdk* in another namespace
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
#endif

    { NULL, 0 }
    };

    for ( Gtk::ConstIntTab* tab = intConstants; tab->name; ++tab )
        mod->addConstant( tab->name, tab->value );
}


} // Gtk
} // Falcon
