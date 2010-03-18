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

    c_RadioButton->setWKS( true );
    c_RadioButton->getClassDef()->factory( &RadioButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_group_changed",       &RadioButton::signal_group_changed },
    { "get_group",                  &RadioButton::get_group },
    { "set_group",                  &RadioButton::set_group },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_RadioButton, meth->name, meth->cb );
}


RadioButton::RadioButton( const Falcon::CoreClass* gen, const GtkRadioButton* btn )
    :
    Gtk::CoreGObject( gen )
{
    if ( btn )
        setUserData( new GData( (GObject*) btn ) );
}


Falcon::CoreObject* RadioButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new RadioButton( gen, (GtkRadioButton*) btn );
}


/*#
    @class gtk.RadioButton
    @brief A choice from multiple check buttons
    @optparam label (string)
    @optparam mnemonic (boolean, default false)
    @optparam group_member (another gtk.RadioButton belonging to that group)

    A single radio button performs the same basic function as a gtk.CheckButton, as
    its position in the object hierarchy reflects. It is only when multiple radio
    buttons are grouped together that they become a different user interface component
    in their own right.

    Every radio button is a member of some group of radio buttons. When one is selected,
    all other radio buttons in the same group are deselected. A gtk.RadioButton is
    one way of giving the user a choice from many options.

    When an unselected button in the group is clicked the clicked button receives the
    "toggled" signal, as does the previously selected button. Inside the "toggled"
    handler, get_active() can be used to determine if the button has been selected
    or deselected.
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

    Emitted when the group of radio buttons that a radio button belongs to changes.
    This is emitted when a radio button switches from being alone to being part of
    a group of 2 or more buttons, or vice-versa, and when a button is moved from
    one group of 2 or more buttons to a different one, but not when the composition
    of the group that a button belongs to changes.
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


/*#
    @method get_group gtk.RadioButton
    @brief Retrieves the group assigned to a radio button.
    @return array containing gtk.RadioButton objects belonging to the group.
 */
FALCON_FUNC RadioButton::get_group( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    CoreArray* arr = new CoreArray;
    GSList* grp = gtk_radio_button_get_group( (GtkRadioButton*)_obj );
    if ( grp && grp->data )
    {
        Item* wki = vm->findWKI( "RadioButton" );
        for ( ; grp; grp = grp->next )
        {
            arr->append( new RadioButton( wki->asClass(), (GtkRadioButton*) grp->data ) );
        }
    }
    vm->retval( arr );
}


/*#
    @method set_group gtk.RadioButton
    @brief Sets a gtk.RadioButton's group.
    @param group_member a gtk.RadioButton belonging to the group

    It should be noted that this does not change the layout of your interface in
    any way, so if you are changing the group, it is likely you will need to re-arrange
    the user interface to reflect these changes.
 */
FALCON_FUNC RadioButton::set_group( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() ||
        !( i_wdt->isOfClass( "RadioButton" ) || i_wdt->isOfClass( "gtk.RadioButton" ) ) )
        throw_inv_params( "RadioButton" );
#endif
    GtkRadioButton* wdt = (GtkRadioButton*)((GData*)i_wdt->asObject()->getUserData())->obj();
    GSList* grp = gtk_radio_button_get_group( wdt );
    if ( !grp || !grp->data ) // just in case..
        throw_inv_params( "Group not found" );
    MYSELF;
    GET_OBJ( self );
    gtk_radio_button_set_group( (GtkRadioButton*)_obj, grp );
}


} // Gtk
} // Falcon
