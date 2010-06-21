/**
 *  \file gtk_EventBox.cpp
 */

#include "gtk_EventBox.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void EventBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_EventBox = mod->addClass( "GtkEventBox", &EventBox::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_EventBox->getClassDef()->addInheritance( in );

    c_EventBox->getClassDef()->factory( &EventBox::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_above_child",        &EventBox::set_above_child },
    { "get_above_child",        &EventBox::get_above_child },
    { "set_visible_window",     &EventBox::set_visible_window },
    { "get_visible_window",     &EventBox::get_visible_window },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_EventBox, meth->name, meth->cb );
}


EventBox::EventBox( const Falcon::CoreClass* gen, const GtkEventBox* box )
    :
    Gtk::CoreGObject( gen, (GObject*) box )
{}


Falcon::CoreObject* EventBox::factory( const Falcon::CoreClass* gen, void* box, bool )
{
    return new EventBox( gen, (GtkEventBox*) box );
}


/*#
    @class GtkEventBox
    @brief A widget used to catch events for widgets which do not have their own window

    The GtkEventBox widget is a subclass of GtkBin which also has its own window.
    It is useful since it allows you to catch events for widgets which do not have
    their own window.
 */
FALCON_FUNC EventBox::init( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* box = gtk_event_box_new();
    self->setGObject( (GObject*) box );
}


/*#
    @method set_above_child GtkEventBox
    @brief Set whether the event box window is positioned above the windows of its child, as opposed to below it.
    @param above_child (bollean) true if the event box window is above the windows of its child

    If the window is above, all events inside the event box will go to the event box.
    If the window is below, events in windows of child widgets will first got to that
    widget, and then to its parents.

    The default is to keep the window below the child.
 */
FALCON_FUNC EventBox::set_above_child( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_event_box_set_above_child( (GtkEventBox*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_above_child GtkEventBox
    @brief Returns whether the event box window is above or below the windows of its child.
    @return true if the event box window is above the window of its child.

    See set_above_child() for details.
 */
FALCON_FUNC EventBox::get_above_child( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_event_box_get_above_child( (GtkEventBox*)_obj ) );
}


/*#
    @method set_visible_window GtkEventBox
    @brief Set whether the event box uses a visible or invisible child window.
    @param visible_window (boolean)

    The default is to use visible windows.

    In an invisible window event box, the window that the event box creates is a
    GDK_INPUT_ONLY window, which means that it is invisible and only serves to receive events.

    A visible window event box creates a visible (GDK_INPUT_OUTPUT) window that acts
    as the parent window for all the widgets contained in the event box.

    You should generally make your event box invisible if you just want to trap events.
    Creating a visible window may cause artifacts that are visible to the user, especially
    if the user is using a theme with gradients or pixmaps.

    The main reason to create a non input-only event box is if you want to set the
    background to a different color or draw on it.

    Note: There is one unexpected issue for an invisible event box that has its window
    below the child. (See set_above_child().) Since the input-only window
    is not an ancestor window of any windows that descendent widgets of the event box
    create, events on these windows aren't propagated up by the windowing system, but
    only by GTK+. The practical effect of this is if an event isn't in the event mask
    for the descendant window (see gtk_widget_add_events()), it won't be received by
    the event box.

    This problem doesn't occur for visible event boxes, because in that case, the event
    box window is actually the ancestor of the descendant windows, not just at the same
    place on the screen.

 */
FALCON_FUNC EventBox::set_visible_window( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_event_box_set_visible_window( (GtkEventBox*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_visible_window GtkEventBox
    @brief Returns whether the event box has a visible window.
    @return true if the event box window is visible

    See set_visible_window() for details.
 */
FALCON_FUNC EventBox::get_visible_window( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_event_box_get_visible_window( (GtkEventBox*)_obj ) );
}


} // Gtk
} // Falcon
