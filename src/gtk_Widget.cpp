/**
 *  \file gtk_Widget.cpp
 */

#include "gtk_Widget.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {


void Widget::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Widget = mod->addClass( "Widget", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Object" ) );
    c_Widget->getClassDef()->addInheritance( in );

    c_Widget->setWKS( true );
    c_Widget->getClassDef()->factory( &Widget::factory );

    Gtk::MethodTab methods[] =
    {

    { "signal_delete_event",    &Widget::signal_delete_event },
    { "signal_show",            &Widget::signal_show },
    { "signal_hide",            &Widget::signal_hide },

    { "show",                   &Widget::show },
    { "show_now",               &Widget::show_now },
    { "hide",                   &Widget::hide },
    { "show_all",               &Widget::show_all },
    { "hide_all",               &Widget::hide_all },
    { "reparent",               &Widget::reparent },
    { "is_focus",               &Widget::is_focus },
    { "grab_focus",             &Widget::grab_focus },
    { "grab_default",           &Widget::grab_default },
    { "set_name",               &Widget::set_name },
    { "get_name",               &Widget::get_name },
    { "set_sensitive",          &Widget::set_sensitive },
    { "get_toplevel",           &Widget::set_sensitive },
    { "get_events",             &Widget::get_events },
    { "is_ancestor",            &Widget::is_ancestor },
    { "hide_on_delete",         &Widget::hide_on_delete },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Widget, meth->name, meth->cb );
}


Widget::Widget( const Falcon::CoreClass* gen, const GtkWidget* wdt )
    :
    Falcon::CoreObject( gen )
{
    if ( wdt )
        setUserData( new GData( (GObject*) wdt ) );
}

bool Widget::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    return defaultProperty( s, it );
}


bool Widget::setProperty( const Falcon::String&, const Falcon::Item& )
{
    return false;
}


Falcon::CoreObject* Widget::factory( const Falcon::CoreClass* gen, void* wdt, bool )
{
    return new Widget( gen, (GtkWidget*) wdt );
}


/*#
    @class gtk.Widget
    @brief Base class for all widgets
    @raise GtkError on direct instanciation

    GtkWidget is the base class all widgets in GTK+ derive from.
    It manages the widget lifecycle, states and style.
 */

/*#
    @init gtk.Widget
    @brief Base class for all widgets
    @raise GtkError on direct instanciation
 */

/*#
    @method signal_delete_event gtk.Widget
    @brief Connect a VMSlot to the widget delete_event signal and return it

    The callback function must return a boolean, that if is true, will block the event.

    The delete-event signal is emitted if a user requests that a toplevel window
    is closed. The default handler for this signal destroys the window.
    Connecting gtk_widget_hide_on_delete() to this signal will cause the window
    to be hidden instead, so that it can later be shown again without reconstructing it.
 */
FALCON_FUNC Widget::signal_delete_event( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "delete_event", (void*) &Widget::on_delete_event, vm );
}


gboolean Widget::on_delete_event( GtkWidget* obj, GdkEvent* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "delete_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_delete_event", it ) )
            {
                printf(
                "[Widget::on_delete_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        //vm->pushParam( Item( (int64)((GdkEventAny*)ev)->type ) );
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[Widget::on_delete_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


/*#
    @method signal_show gtk.Widget
    @brief Connect a VMSlot to the widget show signal and return it
 */
FALCON_FUNC Widget::signal_show( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "show", (void*) &Widget::on_show, vm );
}


void Widget::on_show( GtkWidget* obj, GdkEvent*, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) obj, "show", "on_show", (VMachine*)_vm );
}


/*#
    @method signal_hide gtk.Widget
    @brief Connect a VMSlot to the widget hide signal and return it
 */
FALCON_FUNC Widget::signal_hide( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "hide", (void*) &Widget::on_hide, vm );
}


void Widget::on_hide( GtkWidget* obj, GdkEvent*, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) obj, "hide", "on_hide", (VMachine*)_vm );
}


/*#
    @method show gtk.Widget
    @brief Flags a widget to be displayed.
 */
FALCON_FUNC Widget::show( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show( ((GtkWidget*)_obj) );
}


/*#
    @method show_now gtk.Widget
    @brief Shows a widget.
 */
FALCON_FUNC Widget::show_now( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show_now( ((GtkWidget*)_obj) );
}


/*#
    @method hide gtk.Widget
    @brief Reverses the effects of show(), causing the widget to be hidden (invisible to the user).
 */
FALCON_FUNC Widget::hide( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_hide( ((GtkWidget*)_obj) );
}


/*#
    @method show_all gtk.Widget
    @brief Recursively shows a widget, and any child widgets (if the widget is a container).
 */
FALCON_FUNC Widget::show_all( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show_all( ((GtkWidget*)_obj) );
}


/*#
    @method hide_all gtk.Widget
    @brief Recursively hides a widget and any child widgets.
 */
FALCON_FUNC Widget::hide_all( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_hide_all( ((GtkWidget*)_obj) );
}


/*#
    @method activate gtk.Widget
    @brief For widgets that can be "activated" (buttons, menu items, etc).
    @return boolean
 */
FALCON_FUNC Widget::activate( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_widget_activate( ((GtkWidget*)_obj) ) );
}


/*#
    @method reparent gtk.Widget
    @brief Moves a widget from one container to another.
    @param new_parent The new parent
 */
FALCON_FUNC Widget::reparent( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() ||
        !( i_wdt->isOfClass( "Widget" ) || i_wdt->isOfClass( "gtk.Widget" ) ) )
    {
        throw_inv_params( "Widget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    gtk_widget_reparent( (GtkWidget*)_obj, (GtkWidget*) wdt );
}


/*#
    @method is_focus gtk.Widget
    @brief Determines if the widget is the focus widget within its toplevel.
    This does not mean that the HAS_FOCUS flag is necessarily set;
    HAS_FOCUS will only be set if the toplevel widget additionally has the
    global input focus.
    @return (boolean)
 */
FALCON_FUNC Widget::is_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_widget_is_focus( (GtkWidget*)_obj ) );
}


/*#
    @method grab_focus gtk.Widget
    @brief Causes widget to have the keyboard focus for the GtkWindow it's inside.
    widget must be a focusable widget, such as a GtkEntry; something like GtkFrame won't work.
    More precisely, it must have the GTK_CAN_FOCUS flag set.
    Use gtk_widget_set_can_focus() to modify that flag.
 */
FALCON_FUNC Widget::grab_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_grab_focus( (GtkWidget*)_obj );
}


/*#
    @method grab_default gtk.Widget
    @brief Causes widget to become the default widget.
    widget must have the GTK_CAN_DEFAULT flag set; typically you have to set this flag
    yourself by calling gtk_widget_set_can_default (widget, TRUE). The default widget
    is activated when the user presses Enter in a window. Default widgets must be
    activatable, that is, gtk_widget_activate() should affect them.
 */
FALCON_FUNC Widget::grab_default( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_grab_default( (GtkWidget*)_obj );
}


/*#
    @method set_name gtk.Widget
    @brief Attribute a name to the widget.
    @param name (string)
    Widgets can be named, which allows you to refer to them from a gtkrc file.
    You can apply a style to widgets with a particular name in the gtkrc file.
    See the documentation for gtkrc files (on the same page as the docs for GtkRcStyle).
    Note that widget names are separated by periods in paths (see gtk_widget_path()),
    so names with embedded periods may cause confusion.
 */
FALCON_FUNC Widget::set_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || i_name->isNil() || !i_name->isString() )
    {
        throw_inv_params( "S" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString s( i_name->asString() );
    gtk_widget_set_name( (GtkWidget*)_obj, s.c_str() );
}


/*#
    @method get_name gtk.Widget
    @brief Get the name of the widget.
    @return (string)
    See gtk.Widget.set_name() for the significance of widget names.
 */
FALCON_FUNC Widget::get_name( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* s = gtk_widget_get_name( (GtkWidget*)_obj );
    vm->retval( String( s ) );
}


/*#
    @method set_sensitive gtk.Widget
    @brief Sets the sensitivity of a widget.
    A widget is sensitive if the user can interact with it.
    Insensitive widgets are "grayed out" and the user can't interact with them.
    Insensitive widgets are known as "inactive", "disabled", or "ghosted" in some other toolkits.
 */
FALCON_FUNC Widget::set_sensitive( VMARG )
{
    Item* i_sens = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sens || i_sens->isNil() || !i_sens->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_set_sensitive( (GtkWidget*)_obj, i_sens->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_toplevel gtk.Widget
    @brief This function returns the topmost widget in the container hierarchy widget is a part of.
    @return (widget)
    If widget has no parent widgets, it will be returned as the topmost widget.
    No reference will be added to the returned widget; it should not be unreferenced.
 */
FALCON_FUNC Widget::get_toplevel( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* gwdt = gtk_widget_get_toplevel( (GtkWidget*)_obj );
    Item* wki = vm->findWKI( "Widget" );
    vm->retval( new Widget( wki->asClass(), gwdt ) );
}


/*#
    @method get_events gtk.Widget
    @brief Returns the event mask for the widget.
    @return (integer)
    (A bitfield containing flags from the GdkEventMask enumeration.)
    These are the events that the widget will receive.
 */
FALCON_FUNC Widget::get_events( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_widget_get_events( (GtkWidget*)_obj ) );
}


/*#
    @method is_ancestor gtk.Widget
    @brief Determines whether widget is somewhere inside ancestor, possibly with intermediate containers.
    @return (boolean)
 */
FALCON_FUNC Widget::is_ancestor( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() ||
        !( i_wdt->isOfClass( "Widget" ) || i_wdt->isOfClass( "gtk.Widget" ) ) )
    {
        throw_inv_params( "Widget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    vm->retval( (bool) gtk_widget_is_ancestor( (GtkWidget*)_obj, wdt ) );
}


/*#
    @method hide_on_delete gtk.Widget
    @brief Utility function.
    @return (boolean) always true
    Intended to be connected to the "delete-event" signal on a GtkWindow.
    The function calls gtk_widget_hide() on its argument, then returns TRUE.
    If connected to ::delete-event, the result is that clicking the close button
    for a window (on the window frame, top right corner usually) will hide but
    not destroy the window.
    By default, GTK+ destroys windows when ::delete-event is received.
 */
FALCON_FUNC Widget::hide_on_delete( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gboolean b = gtk_widget_hide_on_delete( (GtkWidget*)_obj );
    vm->retval( (bool) b );
}


} // Gtk
} // Falcon
