/**
 *  \file gtk_RadioButton.cpp
 */

#include "gtk_RadioButton.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void RadioButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_RadioButton = mod->addClass( "GtkRadioButton", &RadioButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCheckButton" ) );
    c_RadioButton->getClassDef()->addInheritance( in );

    c_RadioButton->setWKS( true );
    c_RadioButton->getClassDef()->factory( &RadioButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_group_changed",       &RadioButton::signal_group_changed },
    { "new_with_label",             &RadioButton::new_with_label },
    { "new_with_mnemonic",          &RadioButton::new_with_mnemonic },
    { "get_group",                  &RadioButton::get_group },
    { "set_group",                  &RadioButton::set_group },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_RadioButton, meth->name, meth->cb );
}


RadioButton::RadioButton( const Falcon::CoreClass* gen, const GtkRadioButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* RadioButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new RadioButton( gen, (GtkRadioButton*) btn );
}


/*#
    @class GtkRadioButton
    @brief A choice from multiple check buttons
    @param radio_group_member an existing GtkRadioButton, or nil if you are creating a new group.

    A single radio button performs the same basic function as a GtkCheckButton, as
    its position in the object hierarchy reflects. It is only when multiple radio
    buttons are grouped together that they become a different user interface component
    in their own right.

    Every radio button is a member of some group of radio buttons. When one is selected,
    all other radio buttons in the same group are deselected. A GtkRadioButton is
    one way of giving the user a choice from many options.

    When an unselected button in the group is clicked the clicked button receives the
    "toggled" signal, as does the previously selected button. Inside the "toggled"
    handler, get_active() can be used to determine if the button has been selected
    or deselected.
 */
FALCON_FUNC RadioButton::init( VMARG )
{
    Item* i_grp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isNil() || ( i_grp->isObject()
        && IS_DERIVED( i_grp, GtkRadioButton ) ) ) )
        throw_inv_params( "[GtkRadioButton]" );
#endif
    GtkRadioButton* grp = i_grp->isNil() ? NULL
                        : (GtkRadioButton*) COREGOBJECT( i_grp )->getObject();
    GtkWidget* btn = grp ? gtk_radio_button_new_from_widget( grp )
                    : gtk_radio_button_new( NULL );
    MYSELF;
    self->setObject( (GObject*) btn );
}


/*#
    @method signal_group_changed GtkRadioButton
    @brief Emitted when the group of radio buttons that a radio button belongs to changes.

    This is emitted when a radio button switches from being alone to being part of
    a group of 2 or more buttons, or vice-versa, and when a button is moved from
    one group of 2 or more buttons to a different one, but not when the composition
    of the group that a button belongs to changes.
 */
FALCON_FUNC RadioButton::signal_group_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "group_changed", (void*) &RadioButton::on_group_changed, vm );
}


void RadioButton::on_group_changed( GtkRadioButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "group_changed",
        "on_group_changed", (VMachine*)_vm );
}


/*#
    @method new_with_label GtkRadioButton
    @brief Creates a new GtkRadioButton with a text label, adding it to the same group as radio_group_member.
    @param radio_group_member widget to get radio group from, or nil if you are creating a new group.
    @param label the text label to display next to the radio button.
 */
FALCON_FUNC RadioButton::new_with_label( VMARG )
{
    Item* i_grp = vm->param( 0 );
    Item* i_lbl = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isNil() || ( i_grp->isObject()
        && IS_DERIVED( i_grp, GtkRadioButton ) ) )
        || !i_lbl || !i_lbl->isString() )
        throw_inv_params( "[GtkRadioButton],S" );
#endif
    GtkRadioButton* grp = i_grp->isNil() ? NULL
                    : (GtkRadioButton*) COREGOBJECT( i_grp )->getObject();
    AutoCString lbl( i_lbl->asString() );
    GtkWidget* btn = grp ? gtk_radio_button_new_with_label_from_widget( grp, lbl.c_str() )
                    : gtk_radio_button_new_with_label( NULL, lbl.c_str() );
    vm->retval( new Gtk::RadioButton( vm->findWKI( "GtkRadioButton" )->asClass(),
                                      (GtkRadioButton*) btn ) );
}


/*#
    @method new_with_mnemonic GtkRadioButton
    @brief Creates a new GtkRadioButton containing a label.
    @param radio_group_member widget to get radio group from, or nil if you are creating a new group.
    @param label the text of the button, with an underscore in front of the mnemonic character

    The label will be created using gtk_label_new_with_mnemonic(), so underscores
    in label indicate the mnemonic for the button.
 */
FALCON_FUNC RadioButton::new_with_mnemonic( VMARG )
{
    Item* i_grp = vm->param( 0 );
    Item* i_lbl = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isNil() || ( i_grp->isObject()
        && IS_DERIVED( i_grp, GtkRadioButton ) ) )
        || !i_lbl || !i_lbl->isString() )
        throw_inv_params( "[GtkRadioButton],S" );
#endif
    GtkRadioButton* grp = i_grp->isNil() ? NULL
                    : (GtkRadioButton*) COREGOBJECT( i_grp )->getObject();
    AutoCString lbl( i_lbl->asString() );
    GtkWidget* btn = grp ? gtk_radio_button_new_with_mnemonic_from_widget( grp, lbl.c_str() )
                    : gtk_radio_button_new_with_mnemonic( NULL, lbl.c_str() );
    vm->retval( new Gtk::RadioButton( vm->findWKI( "GtkRadioButton" )->asClass(),
                                      (GtkRadioButton*) btn ) );
}


/*#
    @method get_group GtkRadioButton
    @brief Retrieves the group assigned to a radio button.
    @return an array containing GtkRadioButton objects belonging to the group.
 */
FALCON_FUNC RadioButton::get_group( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GSList* grp = gtk_radio_button_get_group( (GtkRadioButton*)_obj );
    GSList* el;
    int cnt = 0;
    for ( el = grp; el; el = el->next ) ++cnt;
    CoreArray* arr = new CoreArray( cnt );
    if ( cnt )
    {
        Item* wki = vm->findWKI( "GtkRadioButton" );
        for ( el = grp; el; el = el->next )
            arr->append( new Gtk::RadioButton( wki->asClass(), (GtkRadioButton*) grp->data ) );
    }
    vm->retval( arr );
}


/*#
    @method set_group GtkRadioButton
    @brief Sets a GtkRadioButton's group.
    @param group_member a GtkRadioButton belonging to the group

    It should be noted that this does not change the layout of your interface in
    any way, so if you are changing the group, it is likely you will need to re-arrange
    the user interface to reflect these changes.
 */
FALCON_FUNC RadioButton::set_group( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !i_wdt->isObject() || !IS_DERIVED( i_wdt, GtkRadioButton ) )
        throw_inv_params( "GtkRadioButton" );
#endif
    GtkRadioButton* wdt = (GtkRadioButton*) COREGOBJECT( i_wdt )->getObject();
    GSList* grp = gtk_radio_button_get_group( wdt );
    assert( grp && grp->data );
    MYSELF;
    GET_OBJ( self );
    gtk_radio_button_set_group( (GtkRadioButton*)_obj, grp );
}


} // Gtk
} // Falcon
