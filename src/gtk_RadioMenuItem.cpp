/**
 *  \file gtk_RadioMenuItem.cpp
 */

#include "gtk_RadioMenuItem.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void RadioMenuItem::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_RadioMenuItem = mod->addClass( "GtkRadioMenuItem", &RadioMenuItem::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCheckMenuItem" ) );
    c_RadioMenuItem->getClassDef()->addInheritance( in );

    c_RadioMenuItem->setWKS( true );
    c_RadioMenuItem->getClassDef()->factory( &RadioMenuItem::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_group_changed",   &RadioMenuItem::signal_group_changed },
    { "new_with_label",         &RadioMenuItem::new_with_label },
    { "new_with_mnemonic",      &RadioMenuItem::new_with_mnemonic },
#if 0
    { "set_group",              &RadioMenuItem::set_group },
    { "get_group",              &RadioMenuItem::get_group },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_RadioMenuItem, meth->name, meth->cb );
}


RadioMenuItem::RadioMenuItem( const Falcon::CoreClass* gen, const GtkRadioMenuItem* itm )
    :
    Gtk::CoreGObject( gen, (GObject*) itm )
{}


Falcon::CoreObject* RadioMenuItem::factory( const Falcon::CoreClass* gen, void* itm, bool )
{
    return new RadioMenuItem( gen, (GtkRadioMenuItem*) itm );
}


/*#
    @class GtkRadioMenuItem
    @brief A choice from multiple check menu items
    @param group An existing GtkRadioMenuItem, or nil to make a new group of items.

    A radio menu item is a check menu item that belongs to a group. At each instant
    exactly one of the radio menu items from a group is selected.

    [...]
 */
FALCON_FUNC RadioMenuItem::init( VMARG )
{
    Item* i_grp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isNil() || ( i_grp->isObject()
        && IS_DERIVED( i_grp, GtkRadioMenuItem ) ) ) )
        throw_inv_params( "[GtkRadioMenuItem]" );
#endif
    GtkRadioMenuItem* grp = i_grp->isNil() ? NULL
                        : (GtkRadioMenuItem*) COREGOBJECT( i_grp )->getGObject();
    GtkWidget* itm = grp ? gtk_radio_menu_item_new_from_widget( grp )
                        : gtk_radio_menu_item_new( NULL );
    MYSELF;
    self->setGObject( (GObject*) itm );
}


/*#
    @method signal_group_changed GtkRadioMenuItem
    @brief .
 */
FALCON_FUNC RadioMenuItem::signal_group_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "group_changed", (void*) &RadioMenuItem::on_group_changed, vm );
}


void RadioMenuItem::on_group_changed( GtkRadioMenuItem* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "group_changed",
        "on_group_changed", (VMachine*)_vm );
}


/*#
    @method new_with_label GtkRadioMenuItem
    @brief Creates a new GtkRadioMenuItem whose child is a simple GtkLabel.
    @param group an existing GtkRadioMenuItem, or nil to make a new group of items.
    @param label the text for the label
    @return a new GtkRadioMenuItem

    The new GtkRadioMenuItem is added to the same group as group.
 */
FALCON_FUNC RadioMenuItem::new_with_label( VMARG )
{
    const char* spec = "[GtkRadioMenuItem],S";
    Gtk::ArgCheck1 args( vm, spec );
    CoreGObject* o_grp = args.getCoreGObject( 0, false );
    const gchar* lbl = args.getCString( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( o_grp && !CoreObject_IS_DERIVED( o_grp, GtkRadioMenuItem ) )
        throw_inv_params( spec );
#endif
    GtkRadioMenuItem* grp = o_grp ? (GtkRadioMenuItem*) o_grp->getGObject() : NULL;
    GtkWidget* itm = grp ?
        gtk_radio_menu_item_new_with_label_from_widget( grp, lbl )
        : gtk_radio_menu_item_new_with_label( NULL, lbl );
    vm->retval( new Gtk::RadioMenuItem(
        vm->findWKI( "GtkRadioMenuItem" )->asClass(), (GtkRadioMenuItem*) itm ) );
}


/*#
    @method new_with_mnemonic GtkRadioMenuItem
    @brief Creates a new GtkRadioMenuItem containing a label.
    @param group An existing GtkRadioMenuItem, or nil to make a new group of items.
    @param label the text of the button, with an underscore in front of the mnemonic character
    @return a new GtkRadioMenuItem

    The label will be created using gtk_label_new_with_mnemonic(), so underscores
    in label indicate the mnemonic for the menu item.

    The new GtkRadioMenuItem is added to the same group as group.
 */
FALCON_FUNC RadioMenuItem::new_with_mnemonic( VMARG )
{
    const char* spec = "[GtkRadioMenuItem],S";
    Gtk::ArgCheck1 args( vm, spec );
    CoreGObject* o_grp = args.getCoreGObject( 0, false );
    const gchar* lbl = args.getCString( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( o_grp && !CoreObject_IS_DERIVED( o_grp, GtkRadioMenuItem ) )
        throw_inv_params( spec );
#endif
    GtkRadioMenuItem* grp = o_grp ? (GtkRadioMenuItem*) o_grp->getGObject() : NULL;
    GtkWidget* itm = grp ?
        gtk_radio_menu_item_new_with_mnemonic_from_widget( grp, lbl )
        : gtk_radio_menu_item_new_with_mnemonic( NULL, lbl );
    vm->retval( new Gtk::RadioMenuItem(
        vm->findWKI( "GtkRadioMenuItem" )->asClass(), (GtkRadioMenuItem*) itm ) );
}


#if 0
FALCON_FUNC RadioMenuItem::set_group( VMARG );
FALCON_FUNC RadioMenuItem::get_group( VMARG );
#endif

} // Gtk
} // Falcon
