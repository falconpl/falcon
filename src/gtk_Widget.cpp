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

    mod->addClassMethod( c_Widget, "signal_delete_event",   &Widget::signal_delete_event );
    mod->addClassMethod( c_Widget, "signal_show",   &Widget::signal_show );
    mod->addClassMethod( c_Widget, "signal_hide",   &Widget::signal_hide );

    mod->addClassMethod( c_Widget, "show",      &Widget::show );
    mod->addClassMethod( c_Widget, "show_now",  &Widget::show_now );
    mod->addClassMethod( c_Widget, "hide",      &Widget::hide );
    mod->addClassMethod( c_Widget, "show_all",  &Widget::show_all );
    mod->addClassMethod( c_Widget, "hide_all",  &Widget::hide_all );
}


/*#
    @class gtk.Widget
    @brief Abstract class
    @raise GtkError on direct instanciation

 */

/*#
    @init gtk.Widget
    @brief Abstract class
    @raise GtkError on direct instanciation
 */

/*#
    @method signal_delete_event gtk.Widget
    @brief Connect a VMSlot to the widget delete_event signal and return it
    The callback function must return a boolean, that if is true, will block the event.
 */
FALCON_FUNC Widget::signal_delete_event( VMARG )
{
    MYSELF;
    GET_OBJ( self );
    GET_SIGNALS( _obj );
    CoreSlot* cs = get_signal( _obj, _signals,
        "delete_event", (void*) &Widget::on_delete_event, vm );

    Item* it = vm->findWKI( "VMSlot" );
    vm->retval( it->asClass()->createInstance( cs ) );
}


gboolean Widget::on_delete_event( GtkWidget* obj, GdkEvent* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "delete_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item* it;

    do
    {
        it = &iter.getCurrent();

        if ( !it->isCallable()
            && ( it->isComposed()
                && !it->asObject()->getMethod( "on_delete_event", *it ) ) )
        {
            vm->stdErr()->writeString(
            "[Widget::on_delete_event] invalid callback (expected callable)\n" );
            return TRUE; // block event
        }

        //vm->pushParam( Item( (int64)((GdkEventAny*)ev)->type ) );
        vm->callItem( *it, 0 );
        it = &vm->regA();

        if ( notNil( it ) && it->isBoolean() )
        {
            if ( it->asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            vm->stdErr()->writeString(
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
    MYSELF;
    GET_OBJ( self );
    GET_SIGNALS( _obj );
    CoreSlot* ev = get_signal( _obj, _signals,
        "show", (void*) &Widget::on_show, vm );

    Item* it = vm->findWKI( "VMSlot" );
    vm->retval( it->asClass()->createInstance( ev ) );
}


void Widget::on_show( GtkWidget* obj, GdkEvent*, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "show", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item* it;

    do
    {
        it = &iter.getCurrent();
        if ( !it->isCallable()
            && ( it->isComposed()
                && !it->asObject()->getMethod( "on_show", *it ) ) )
        {
            vm->stdErr()->writeString(
            "[Widget::on_show] invalid callback (expected callable)\n" );
            return;
        }
        vm->callItem( *it, 0 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_hide gtk.Widget
    @brief Connect a VMSlot to the widget hide signal and return it
 */
FALCON_FUNC Widget::signal_hide( VMARG )
{
    MYSELF;
    GET_OBJ( self );
    GET_SIGNALS( _obj );
    CoreSlot* ev = get_signal( _obj, _signals,
        "hide", (void*) &Widget::on_hide, vm );

    Item* it = vm->findWKI( "VMSlot" );
    vm->retval( it->asClass()->createInstance( ev ) );
}


void Widget::on_hide( GtkWidget* obj, GdkEvent*, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "hide", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item* it;

    do
    {
        it = &iter.getCurrent();
        if ( !it->isCallable()
            && ( it->isComposed()
                && !it->asObject()->getMethod( "on_hide", *it ) ) )
        {
            vm->stdErr()->writeString(
            "[Widget::on_hide] invalid callback (expected callable)\n" );
            return;
        }
        vm->callItem( *it, 0 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method show gtk.Widget
    @brief Flags a widget to be displayed.
 */
FALCON_FUNC Widget::show( VMARG )
{
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }

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
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }

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
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }

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
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }

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
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }

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
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }

    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_widget_activate( ((GtkWidget*)_obj) ) );
}


} // Gtk
} // Falcon
