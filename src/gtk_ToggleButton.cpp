/**
 *  \file gtk_ToggleButton.cpp
 */

#include "gtk_ToggleButton.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ToggleButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ToggleButton = mod->addClass( "ToggleButton", &ToggleButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Button" ) );
    c_ToggleButton->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    { "signal_toggled",         &ToggleButton::signal_toggled },
    { "set_mode",               &ToggleButton::set_mode },
    { "get_mode",               &ToggleButton::get_mode },
    { "toggled",                &ToggleButton::toggled },
    { "get_active",             &ToggleButton::get_active },
    { "set_active",             &ToggleButton::set_active },
    { "get_inconsistent",       &ToggleButton::get_inconsistent },
    { "set_inconsistent",       &ToggleButton::set_inconsistent },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ToggleButton, meth->name, meth->cb );
}

/*#
    @class gtk.ToggleButton
    @brief Create buttons which retain their state
    @optparam label (string)
    @optparam mnemonic (boolean, default false)

    A GtkToggleButton is a GtkButton which will remain 'pressed-in' when clicked.
    Clicking again will cause the toggle button to return to its normal state.

    The state of a GtkToggleButton can be set specifically using set_active(),
    and retrieved using get_active().

    To simply switch the state of a toggle button, use toggled().
 */
FALCON_FUNC ToggleButton::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
        return;

    Item* i_lbl = vm->param( 0 );
    Item* i_mne = vm->param( 1 );
    GtkWidget* btn;

    if ( i_lbl )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_lbl->isNil() || !i_lbl->isString() )
            throw_inv_params( "[S,B]" );
#endif
        AutoCString lbl( i_lbl->asString() );

        if ( i_mne )
        {
#ifndef NO_PARAMETER_CHECK
            if ( i_mne->isNil() || !i_mne->isBoolean() )
                throw_inv_params( "[S,B]" );
#endif
            if ( i_mne->asBoolean() )
                btn = gtk_toggle_button_new_with_mnemonic( lbl.c_str() );
            else
                btn = gtk_toggle_button_new_with_label( lbl.c_str() );
        }
        else
            btn = gtk_toggle_button_new_with_label( lbl.c_str() );
    }
    else
        btn = gtk_toggle_button_new();

    Gtk::internal_add_slot( (GObject*) btn );
    self->setUserData( new GData( (GObject*) btn ) );
}


/*#
    @method signal_toggled gtk.ToggleButton
    @brief Connect a VMSlot to the button toggled signal and return it

    Should be connected if you wish to perform an action whenever the
    GtkToggleButton's state is changed.
 */
FALCON_FUNC ToggleButton::signal_toggled( VMARG )
{
    Gtk::internal_get_slot( "toggled", (void*) &ToggleButton::on_toggled, vm );
}


void ToggleButton::on_toggled( GtkToggleButton* btn, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) btn, "toggled", "on_toggled", (VMachine*)_vm );
}


/*#
    @method set_mode gtk.ToggleButton
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
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_button_set_mode( (GtkToggleButton*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_mode gtk.ToggleButton
    @brief Retrieves whether the button is displayed as a separate indicator and label.
    @return (boolean) true if the togglebutton is drawn as a separate indicator and label.
 */
FALCON_FUNC ToggleButton::get_mode( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_button_get_mode( (GtkToggleButton*)_obj ) );
}


/*#
    @method toggled gtk.ToggleButton
    @brief Emits the toggled signal on the GtkToggleButton.

    There is no good reason for an application ever to call this function.
 */
FALCON_FUNC ToggleButton::toggled( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_button_toggled( (GtkToggleButton*)_obj );
}


/*#
    @method get_active gtk.ToggleButton
    @brief Queries a GtkToggleButton and returns its current state.
    @return true if the toggle button is pressed in and false if it is raised.
 */
FALCON_FUNC ToggleButton::get_active( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_button_get_active( (GtkToggleButton*)_obj ) );
}


/*#
    @method set_active gtk.ToggleButton
    @brief Sets the status of the toggle button.
    @param is_active Set to true if you want the GtkToggleButton to be 'pressed in', and false to raise it.

    This action causes the toggled signal to be emitted.
 */
FALCON_FUNC ToggleButton::set_active( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_button_set_active( (GtkToggleButton*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_inconsistent gtk.ToggleButton
    @brief Gets the value set by set_inconsistent().
    @return (boolean) true if the button is displayed as inconsistent, false otherwise
 */
FALCON_FUNC ToggleButton::get_inconsistent( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_button_get_inconsistent( (GtkToggleButton*)_obj ) );
}


/*#
    @method set_inconsistent gtk.ToggleButton
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
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_button_set_inconsistent( (GtkToggleButton*)_obj,
        i_bool->asBoolean() ? TRUE : FALSE );
}


} // Gtk
} // Falcon
