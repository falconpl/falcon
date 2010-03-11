/**
 *  \file gtk_enums.cpp
 */

#include "gtk_enums.hpp"

#include <gtk/gtk.h>

namespace Falcon {
namespace Gtk {

void Enums::modInit( Falcon::Module* mod )
{
/*
 *  GtkWidgetFlags
 */
mod->addConstant( "TOPLEVEL",       (Falcon::int64) GTK_TOPLEVEL );
mod->addConstant( "NO_WINDOW",      (Falcon::int64) GTK_NO_WINDOW );
mod->addConstant( "REALIZED",       (Falcon::int64) GTK_REALIZED );
mod->addConstant( "MAPPED",         (Falcon::int64) GTK_MAPPED );
mod->addConstant( "VISIBLE",        (Falcon::int64) GTK_VISIBLE );
mod->addConstant( "SENSITIVE",      (Falcon::int64) GTK_SENSITIVE );
mod->addConstant( "PARENT_SENSITIVE",(Falcon::int64) GTK_PARENT_SENSITIVE );
mod->addConstant( "CAN_FOCUS",      (Falcon::int64) GTK_CAN_FOCUS );
mod->addConstant( "HAS_FOCUS",      (Falcon::int64) GTK_HAS_FOCUS );
mod->addConstant( "CAN_DEFAULT",    (Falcon::int64) GTK_CAN_DEFAULT );
mod->addConstant( "HAS_DEFAULT",    (Falcon::int64) GTK_HAS_DEFAULT );
mod->addConstant( "HAS_GRAB",       (Falcon::int64) GTK_HAS_GRAB );
mod->addConstant( "RC_STYLE",       (Falcon::int64) GTK_RC_STYLE );
mod->addConstant( "COMPOSITE_CHILD",(Falcon::int64) GTK_COMPOSITE_CHILD );
mod->addConstant( "NO_REPARENT",    (Falcon::int64) GTK_NO_REPARENT );
mod->addConstant( "APP_PAINTABLE",  (Falcon::int64) GTK_APP_PAINTABLE );
mod->addConstant( "RECEIVES_DEFAULT",(Falcon::int64) GTK_RECEIVES_DEFAULT );
mod->addConstant( "DOUBLE_BUFFERED",(Falcon::int64) GTK_DOUBLE_BUFFERED );
mod->addConstant( "NO_SHOW_ALL",    (Falcon::int64) GTK_NO_SHOW_ALL );

/*
 *  GtkWindowType
 */
mod->addConstant( "WINDOW_TOPLEVEL",    (Falcon::int64) GTK_WINDOW_TOPLEVEL );
mod->addConstant( "WINDOW_POPUP",       (Falcon::int64) GTK_WINDOW_POPUP );

#if 0
/*
 *  GdkEventType
 *  we might put Gdk* in another namespace
 */
mod->addConstant( "GDK_NOTHING",        (Falcon::int64) GDK_NOTHING );
mod->addConstant( "GDK_DELETE",         (Falcon::int64) GDK_DELETE );
mod->addConstant( "GDK_DESTROY",        (Falcon::int64) GDK_DESTROY );
mod->addConstant( "GDK_EXPOSE",         (Falcon::int64) GDK_EXPOSE );
mod->addConstant( "GDK_MOTION_NOTIFY",  (Falcon::int64) GDK_MOTION_NOTIFY );
mod->addConstant( "GDK_BUTTON_PRESS",   (Falcon::int64) GDK_BUTTON_PRESS );
mod->addConstant( "GDK_2BUTTON_PRESS",  (Falcon::int64) GDK_2BUTTON_PRESS );
mod->addConstant( "GDK_3BUTTON_PRESS",  (Falcon::int64) GDK_3BUTTON_PRESS );
mod->addConstant( "GDK_BUTTON_RELEASE", (Falcon::int64) GDK_BUTTON_RELEASE );
mod->addConstant( "GDK_KEY_PRESS",      (Falcon::int64) GDK_KEY_PRESS );
mod->addConstant( "GDK_KEY_RELEASE",    (Falcon::int64) GDK_KEY_RELEASE );
mod->addConstant( "GDK_ENTER_NOTIFY",   (Falcon::int64) GDK_ENTER_NOTIFY );
mod->addConstant( "GDK_LEAVE_NOTIFY",   (Falcon::int64) GDK_LEAVE_NOTIFY );
mod->addConstant( "GDK_FOCUS_CHANGE",   (Falcon::int64) GDK_FOCUS_CHANGE );
mod->addConstant( "GDK_CONFIGURE",      (Falcon::int64) GDK_CONFIGURE );
mod->addConstant( "GDK_MAP",            (Falcon::int64) GDK_MAP );
mod->addConstant( "GDK_UNMAP",          (Falcon::int64) GDK_UNMAP );
mod->addConstant( "GDK_PROPERTY_NOTIFY",(Falcon::int64) GDK_PROPERTY_NOTIFY );
mod->addConstant( "GDK_SELECTION_CLEAR",(Falcon::int64) GDK_SELECTION_CLEAR );
mod->addConstant( "GDK_SELECTION_REQUEST",(Falcon::int64) GDK_SELECTION_REQUEST );
mod->addConstant( "GDK_SELECTION_NOTIFY",(Falcon::int64) GDK_SELECTION_NOTIFY );
mod->addConstant( "GDK_PROXIMITY_IN",   (Falcon::int64) GDK_PROXIMITY_IN );
mod->addConstant( "GDK_PROXIMITY_OUT",  (Falcon::int64) GDK_PROXIMITY_OUT );
mod->addConstant( "GDK_DRAG_ENTER",     (Falcon::int64) GDK_DRAG_ENTER );
mod->addConstant( "GDK_DRAG_LEAVE",     (Falcon::int64) GDK_DRAG_LEAVE );
mod->addConstant( "GDK_DRAG_MOTION",    (Falcon::int64) GDK_DRAG_MOTION );
mod->addConstant( "GDK_DRAG_STATUS",    (Falcon::int64) GDK_DRAG_STATUS );
mod->addConstant( "GDK_DROP_START",     (Falcon::int64) GDK_DROP_START );
mod->addConstant( "GDK_DROP_FINISHED",  (Falcon::int64) GDK_DROP_FINISHED );
mod->addConstant( "GDK_CLIENT_EVENT",   (Falcon::int64) GDK_CLIENT_EVENT );
mod->addConstant( "GDK_VISIBILITY_NOTIFY",(Falcon::int64) GDK_VISIBILITY_NOTIFY );
mod->addConstant( "GDK_NO_EXPOSE",      (Falcon::int64) GDK_NO_EXPOSE );
mod->addConstant( "GDK_SCROLL",         (Falcon::int64) GDK_SCROLL );
mod->addConstant( "GDK_WINDOW_STATE",   (Falcon::int64) GDK_WINDOW_STATE );
mod->addConstant( "GDK_SETTING",        (Falcon::int64) GDK_SETTING );
#if GTK_MINOR_VERSION >= 6
mod->addConstant( "GDK_OWNER_CHANGE",   (Falcon::int64) GDK_OWNER_CHANGE );
#endif
#if GTK_MINOR_VERSION >= 8
mod->addConstant( "GDK_GRAB_BROKEN",    (Falcon::int64) GDK_GRAB_BROKEN );
#endif
#if GTK_MINOR_VERSION >= 14
mod->addConstant( "GDK_DAMAGE",         (Falcon::int64) GDK_DAMAGE );
#endif
#if GTK_MINOR_VERSION >= 18
mod->addConstant( "GDK_EVENT_LAST",     (Falcon::int64) GDK_EVENT_LAST );
#endif
#endif
}


} // Gtk
} // Falcon
