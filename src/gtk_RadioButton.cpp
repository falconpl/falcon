/**
 *  \file gtk_RadioButton.cpp
 */

#include "gtk_RadioButton.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void RadioButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_RadioButton = mod->addClass( "RadioButton", &RadioButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "CheckButton" ) );
    c_RadioButton->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    { "signal_group_changed",       &RadioButton::signal_group_changed },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_RadioButton, meth->name, meth->cb );
}


/*#
    @class gtk.RadioButton
    @brief A choice from multiple check buttons

    A single radio button performs the same basic function as a gtk.CheckButton, as
    its position in the object hierarchy reflects. It is only when multiple radio
    buttons are grouped together that they become a different user interface component
    in their own right.

    Every radio button is a member of some group of radio buttons. When one is selected,
    all other radio buttons in the same group are deselected. A gtk.RadioButton is
    one way of giving the user a choice from many options.
 */

/*#
    @init gtk.RadioButton
    @brief Creates a new gtk.RadioButton.
    @optparam label (string)
    @optparam mnemonic (boolean, default false)
    @optparam group_member (gtk.RadioButton)
 */
FALCON_FUNC RadioButton::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
        return;

    Item* i_lbl = vm->param( 0 );
    Item* i_mne = vm->param( 1 );
    Item* i_wdt = vm->param( 2 );
    GtkWidget* btn = NULL;

    if ( i_lbl )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_lbl->isNil() || !i_lbl->isString() )
            throw_inv_params( "[S,B,RadioButton]" );
#endif
        AutoCString lbl( i_lbl->asString() );

        if ( i_mne )
        {
#ifndef NO_PARAMETER_CHECK
            if ( i_mne->isNil() || !i_mne->isBoolean() )
                throw_inv_params( "[S,B,RadioButton]" );
#endif
            if ( i_wdt )
            {
#ifndef NO_PARAMETER_CHECK
                if ( i_wdt->isNil()
                    || !( i_wdt->isOfClass( "RadioButton" ) || i_wdt->isOfClass( "gtk.RadioButton" ) ) )
                    throw_inv_params( "[S,B,RadioButton]" );
#endif
                GtkRadioButton* wdt = (GtkRadioButton*)((GData*)i_wdt->asObject()->getUserData())->obj();

                if ( i_mne->asBoolean() )
                    btn = gtk_radio_button_new_with_mnemonic_from_widget( wdt, lbl.c_str() );
                else
                    btn = gtk_radio_button_new_with_label_from_widget( wdt, lbl.c_str() );
            }
            else
            {
                btn = gtk_radio_button_new( NULL );
                gtk_button_set_label( (GtkButton*) btn, lbl.c_str() );
                if ( i_mne->asBoolean() )
                    gtk_button_set_use_underline( (GtkButton*) btn, TRUE );
            }
        }
        else
        {
            btn = gtk_radio_button_new( NULL );
            gtk_button_set_label( (GtkButton*) btn, lbl.c_str() );
        }
    }
    else
        btn = gtk_radio_button_new( NULL );

    assert( btn );
    Gtk::internal_add_slot( (GObject*) btn );
    self->setUserData( new GData( (GObject*) btn ) );
}


/*#
    @method signal_group_changed gtk.RadioButton
    @brief Connect a VMSlot to the button group-changed signal and return it
 */
FALCON_FUNC RadioButton::signal_group_changed( VMARG )
{
    Gtk::internal_get_slot( "group_changed", (void*) &RadioButton::on_group_changed, vm );
}


void RadioButton::on_group_changed( GtkRadioButton* btn, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) btn, "group_changed",
        "on_group_changed", (VMachine*)_vm );
}


//FALCON_FUNC get_group( VMARG );

//FALCON_FUNC set_group( VMARG );



} // Gtk
} // Falcon
