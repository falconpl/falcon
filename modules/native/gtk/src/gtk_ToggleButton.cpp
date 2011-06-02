/**
 *  \file gtk_ToggleButton.cpp
 */

#include "gtk_ToggleButton.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ToggleButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ToggleButton = mod->addClass( "GtkToggleButton", &ToggleButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkButton" ) );
    c_ToggleButton->getClassDef()->addInheritance( in );

    c_ToggleButton->setWKS( true );
    c_ToggleButton->getClassDef()->factory( &ToggleButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_toggled",     &ToggleButton::signal_toggled },
    { "new_with_label",     &ToggleButton::new_with_label },
    { "new_with_mnemonic",  &ToggleButton::new_with_mnemonic },
    { "set_mode",           &ToggleButton::set_mode },
    { "get_mode",           &ToggleButton::get_mode },
    { "toggled",            &ToggleButton::toggled },
    { "get_active",         &ToggleButton::get_active },
    { "set_active",         &ToggleButton::set_active },
    { "get_inconsistent",   &ToggleButton::get_inconsistent },
    { "set_inconsistent",   &ToggleButton::set_inconsistent },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ToggleButton, meth->name, meth->cb );
}


ToggleButton::ToggleButton( const Falcon::CoreClass* gen, const GtkToggleButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* ToggleButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new ToggleButton( gen, (GtkToggleButton*) btn );
}


/*#
    @class GtkToggleButton
    @brief Create buttons which retain their state

    A GtkToggleButton is a GtkButton which will remain 'pressed-in' when clicked.
    Clicking again will cause the toggle button to return to its normal state.

    The state of a GtkToggleButton can be set specifically using set_active(),
    and retrieved using get_active().

    To simply switch the state of a toggle button, use toggled().
 */
FALCON_FUNC ToggleButton::init( VMARG )
{
    MYSELF;
    if ( self->getObject() )
        return;
    NO_ARGS
    self->setObject( (GObject*) gtk_toggle_button_new() );
}


/*#
    @method signal_toggled GtkToggleButton
    @brief Should be connected if you wish to perform an action whenever the GtkToggleButton's state is changed.
 */
FALCON_FUNC ToggleButton::signal_toggled( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "toggled", (void*) &ToggleButton::on_toggled, vm );
}


void ToggleButton::on_toggled( GtkToggleButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "toggled", "on_toggled", (VMachine*)_vm );
}


/*#
    @method new_with_label GtkToggleButton
    @brief Creates a new toggle button with a text label.
    @param label a string containing the message to be placed in the toggle button.
    @return a new toggle button.
 */
FALCON_FUNC ToggleButton::new_with_label( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString lbl( i_lbl->asString() );
    GtkWidget* btn = gtk_toggle_button_new_with_label( lbl.c_str() );
    vm->retval( new Gtk::ToggleButton( vm->findWKI( "GtkToggleButton" )->asClass(),
                                       (GtkToggleButton*) btn ) );
}


/*#
    @method new_with_mnemonic GtkToggleButton
    @brief Creates a new GtkToggleButton containing a label.
    @param label the text of the button, with an underscore in front of the mnemonic character
    @return a new toggle button.

    The label will be created using gtk_label_new_with_mnemonic(), so underscores
    in label indicate the mnemonic for the button.
 */
FALCON_FUNC ToggleButton::new_with_mnemonic( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString lbl( i_lbl->asString() );
    GtkWidget* btn = gtk_toggle_button_new_with_mnemonic( lbl.c_str() );
    vm->retval( new Gtk::ToggleButton( vm->findWKI( "GtkToggleButton" )->asClass(),
                                       (GtkToggleButton*) btn ) );
}


/*#
    @method set_mode GtkToggleButton
    @brief Sets whether the button is displayed as a separate indicator and label.
    @param draw_indicator (boolean) if true, draw the button as a separate indicator and label; if false, draw the button like a normal button

    You can call this function on a checkbutton or a radiobutton with draw_indicator = false
    to make the button look like a normal button

    This function only affects instances of classes like GtkCheckButton and GtkRadioButton
    that derive from GtkToggleButton, not instances of GtkToggleButton itself.
 */
FALCON_FUNC ToggleButton::set_mode( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_button_set_mode( (GtkToggleButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_mode GtkToggleButton
    @brief Retrieves whether the button is displayed as a separate indicator and label.
    @return (boolean) true if the togglebutton is drawn as a separate indicator and label.
 */
FALCON_FUNC ToggleButton::get_mode( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_button_get_mode( (GtkToggleButton*)_obj ) );
}


/*#
    @method toggled GtkToggleButton
    @brief Emits the toggled signal on the GtkToggleButton.

    There is no good reason for an application ever to call this function.
 */
FALCON_FUNC ToggleButton::toggled( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_button_toggled( (GtkToggleButton*)_obj );
}


/*#
    @method get_active GtkToggleButton
    @brief Queries a GtkToggleButton and returns its current state.
    @return true if the toggle button is pressed in and false if it is raised.
 */
FALCON_FUNC ToggleButton::get_active( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_button_get_active( (GtkToggleButton*)_obj ) );
}


/*#
    @method set_active GtkToggleButton
    @brief Sets the status of the toggle button.
    @param is_active Set to true if you want the GtkToggleButton to be 'pressed in', and false to raise it.

    This action causes the toggled signal to be emitted.
 */
FALCON_FUNC ToggleButton::set_active( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_button_set_active( (GtkToggleButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_inconsistent GtkToggleButton
    @brief Gets the value set by set_inconsistent().
    @return (boolean) true if the button is displayed as inconsistent, false otherwise
 */
FALCON_FUNC ToggleButton::get_inconsistent( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_button_get_inconsistent( (GtkToggleButton*)_obj ) );
}


/*#
    @method set_inconsistent GtkToggleButton
    @brief Sets the button consistency state.
    @param setting (boolean) true if state is inconsistent

    If the user has selected a range of elements (such as some text or spreadsheet cells)
    that are affected by a toggle button, and the current values in that range are
    inconsistent, you may want to display the toggle in an "in between" state.
    This function turns on "in between" display. Normally you would turn off the
    inconsistent state again if the user toggles the toggle button. This has to be
    done manually, set_inconsistent() only affects visual appearance, it doesn't affect
    the semantics of the button.
 */
FALCON_FUNC ToggleButton::set_inconsistent( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_button_set_inconsistent( (GtkToggleButton*)_obj,
                                        (gboolean) i_bool->asBoolean() );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
