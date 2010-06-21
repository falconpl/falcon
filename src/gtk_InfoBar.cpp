/**
 *  \file gtk_InfoBar.cpp
 */

#include "gtk_InfoBar.hpp"

#if GTK_MINOR_VERSION >= 18

#include "gtk_Buildable.hpp"
#include "gtk_Orientable.hpp"
#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void InfoBar::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_InfoBar = mod->addClass( "GtkInfoBar", &InfoBar::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkHBox" ) );
    c_InfoBar->getClassDef()->addInheritance( in );

    c_InfoBar->getClassDef()->factory( &InfoBar::factory );

    Gtk::MethodTab methods[] =
    {
    //{ "new_with_buttons",       &InfoBar::new_with_buttons },
    { "add_action_widget",      &InfoBar::add_action_widget },
    { "add_button",             &InfoBar::add_button },
    //{ "add_buttons",            &InfoBar::add_buttons },
    { "set_response_sensitive", &InfoBar::set_response_sensitive },
    { "set_default_response",   &InfoBar::set_default_response },
    { "response",               &InfoBar::response },
    { "set_message_type",       &InfoBar::set_message_type },
    { "get_message_type",       &InfoBar::get_message_type },
    { "get_action_area",        &InfoBar::get_action_area },
    { "get_content_area",       &InfoBar::get_content_area },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_InfoBar, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_InfoBar );
    Gtk::Orientable::clsInit( mod, c_InfoBar );
}


InfoBar::InfoBar( const Falcon::CoreClass* gen, const GtkInfoBar* bar )
    :
    Gtk::CoreGObject( gen, (GObject*) bar )
{}


Falcon::CoreObject* InfoBar::factory( const Falcon::CoreClass* gen, void* bar, bool )
{
    return new InfoBar( gen, (GtkInfoBar*) bar );
}


/*#
    @class GtkInfoBar
    @brief Report important messages to the user

    GtkInfoBar is a widget that can be used to show messages to the user without
    showing a dialog. It is often temporarily shown at the top or bottom of a
    document. In contrast to GtkDialog, which has a horizontal action area at
    the bottom, GtkInfoBar has a vertical action area at the side.

    The API of GtkInfoBar is very similar to GtkDialog, allowing you to add buttons
    to the action area with add_button() or new_with_buttons().
    The sensitivity of action widgets can be controlled with set_response_sensitive().
    To add widgets to the main content area of a GtkInfoBar, use get_content_area()
    and add your widgets to the container.

    Similar to GtkMessageDialog, the contents of a GtkInfoBar can by classified
    as error message, warning, informational message, etc, by using
    set_message_type(). GTK+ uses the message type to determine
    the background color of the message area.

    [...]
 */
FALCON_FUNC InfoBar::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_info_bar_new() );
}


//FALCON_FUNC InfoBar::new_with_buttons( VMARG );


/*#
    @method add_action_widget GtkInfoBar
    @brief Add an activatable widget to the action area of a GtkInfoBar, connecting a signal handler that will emit the "response" signal on the message area when the widget is activated.
    @param child an activatable widget
    @param response_id response ID for child (integer).

    The widget is appended to the end of the message areas action area.
 */
FALCON_FUNC InfoBar::add_action_widget( VMARG )
{
    Item* i_chld = vm->param( 0 );
    Item* i_id = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_chld || i_chld->isNil() || !i_chld->isObject()
        || !IS_DERIVED( i_chld, GtkWidget )
        || !i_id || i_id->isNil() || !i_id->isInteger() )
        throw_inv_params( "GtkWidget,I" );
#endif
    GtkWidget* chld = (GtkWidget*) COREGOBJECT( i_chld )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_info_bar_add_action_widget( (GtkInfoBar*)_obj, chld, i_id->asInteger() );
}


/*#
    @method add_button GtkInfoBar
    @brief Adds a button with the given text (or a stock button, if button_text is a stock ID) and sets things up so that clicking the button will emit the "response" signal with the given response_id.
    @param button_text text of button, or stock ID
    @param response_id response ID for the button
    @return the button widget that was added

    The button is appended to the end of the info bars's action area. The
    button widget is returned, but usually you don't need it.
 */
FALCON_FUNC InfoBar::add_button( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,I" );
    char* txt = args.getCString( 0 );
    int id = args.getInteger( 1 );
    MYSELF;
    GET_OBJ( self );
    GtkWidget* btn = gtk_info_bar_add_button( (GtkInfoBar*)_obj, txt, id );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), btn ) );
}


//FALCON_FUNC InfoBar::add_buttons( VMARG );


/*#
    @method set_response_sensitive GtkInfoBar
    @brief Calls gtk_widget_set_sensitive (widget, setting) for each widget in the info bars's action area with the given response_id.
    @param response_id a response ID
    @param setting true for sensitive

    A convenient way to sensitize/desensitize dialog buttons.
 */
FALCON_FUNC InfoBar::set_response_sensitive( VMARG )
{
    Item* i_id = vm->param( 0 );
    Item* i_bool = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || i_id->isNil() || !i_id->isInteger()
        || !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "I,B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_info_bar_set_response_sensitive( (GtkInfoBar*)_obj,
                    i_id->asInteger(), i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_default_response GtkInfoBar
    @brief Sets the last widget in the info bar's action area with the given response_id as the default widget for the dialog.

    Pressing "Enter" normally activates the default widget.

    Note that this function currently requires info_bar to be added to a widget hierarchy.
 */
FALCON_FUNC InfoBar::set_default_response( VMARG )
{
    Item* i_id = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || i_id->isNil() || !i_id->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_info_bar_set_default_response( (GtkInfoBar*)_obj, i_id->asInteger() );
}


/*#
    @method response GtkInfoBar
    @brief Emits the 'response' signal with the given response_id.
    @param response_id a response ID
 */
FALCON_FUNC InfoBar::response( VMARG )
{
    Item* i_id = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || i_id->isNil() || !i_id->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_info_bar_response( (GtkInfoBar*)_obj, i_id->asInteger() );
}


/*#
    @method set_message_type GtkInfoBar
    @brief Sets the message type of the message area (GtkMessageType).

    GTK+ uses this type to determine what color to use when drawing the message area.
 */
FALCON_FUNC InfoBar::set_message_type( VMARG )
{
    Item* i_type = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_type || i_type->isNil() || !i_type->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_info_bar_set_message_type( (GtkInfoBar*)_obj, (GtkMessageType) i_type->asInteger() );
}


/*#
    @method get_message_type GtkInfoBar
    @brief Returns the message type of the message area.
    @return the message type of the message area.
 */
FALCON_FUNC InfoBar::get_message_type( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_info_bar_get_message_type( (GtkInfoBar*)_obj ) );
}


/*#
    @method get_action_area GtkInfoBar
    @brief Returns the action area of info_bar.
    @return the action area.
 */
FALCON_FUNC InfoBar::get_action_area( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_info_bar_get_action_area( (GtkInfoBar*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}


/*#
    @method get_content_area GtkInfoBar
    @brief Returns the content area of info_bar.
    @return the content area.
 */
FALCON_FUNC InfoBar::get_content_area( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_info_bar_get_content_area( (GtkInfoBar*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}


} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 18
