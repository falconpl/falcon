/**
 *  \file gtk_CheckMenuItem.cpp
 */

#include "gtk_CheckMenuItem.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CheckMenuItem::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CheckMenuItem = mod->addClass( "GtkCheckMenuItem", &CheckMenuItem::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkMenuItem" ) );
    c_CheckMenuItem->getClassDef()->addInheritance( in );

    c_CheckMenuItem->setWKS( true );
    c_CheckMenuItem->getClassDef()->factory( &CheckMenuItem::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_toggled",     &CheckMenuItem::signal_toggled },
    { "new_with_label",     &CheckMenuItem::new_with_label },
    { "new_with_mnemonic",  &CheckMenuItem::new_with_mnemonic },
    //{ "set_state",          &CheckMenuItem::set_state },
    { "get_active",         &CheckMenuItem::get_active },
    { "set_active",         &CheckMenuItem::set_active },
    //{ "set_show_toggle",    &CheckMenuItem::set_show_toggle },
    { "toggled",            &CheckMenuItem::toggled },
    { "get_inconsistent",   &CheckMenuItem::get_inconsistent },
    { "set_inconsistent",   &CheckMenuItem::set_inconsistent },
    { "set_draw_as_radio",  &CheckMenuItem::set_draw_as_radio },
    { "get_draw_as_radio",  &CheckMenuItem::get_draw_as_radio },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_CheckMenuItem, meth->name, meth->cb );
}


CheckMenuItem::CheckMenuItem( const Falcon::CoreClass* gen, const GtkCheckMenuItem* menu )
    :
    Gtk::CoreGObject( gen, (GObject*) menu )
{}


Falcon::CoreObject* CheckMenuItem::factory( const Falcon::CoreClass* gen, void* menu, bool )
{
    return new CheckMenuItem( gen, (GtkCheckMenuItem*) menu );
}


/*#
    @class GtkCheckMenuItem
    @brief A menu item with a check box

    A GtkCheckMenuItem is a menu item that maintains the state of a boolean value
    in addition to a GtkMenuItem's usual role in activating application code.

    A check box indicating the state of the boolean value is displayed at the left
    side of the GtkMenuItem. Activating the GtkMenuItem toggles the value.
 */
FALCON_FUNC CheckMenuItem::init( VMARG )
{
    MYSELF;
    if ( self->getGObject() )
        return;
    NO_ARGS
    self->setGObject( (GObject*) gtk_check_menu_item_new() );
}


/*#
    @method signal_toggled GtkCheckMenuItem
    @brief This signal is emitted when the state of the check box is changed.
 */
FALCON_FUNC CheckMenuItem::signal_toggled( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "toggled", (void*) &CheckMenuItem::on_toggled, vm );
}


void CheckMenuItem::on_toggled( GtkMenuItem* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "toggled", "on_toggled", (VMachine*)_vm );
}


/*#
    @method new_with_label GtkCheckMenuItem
    @brief Creates a new GtkCheckMenuItem with a label.
    @param label the string to use for the label.
    @return a new GtkCheckMenuItem.
 */
FALCON_FUNC CheckMenuItem::new_with_label( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* label = args.getCString( 0 );
    GtkWidget* wdt = gtk_check_menu_item_new_with_label( label );
    vm->retval( new Gtk::CheckMenuItem( vm->findWKI( "GtkCheckMenuItem" )->asClass(),
                                        (GtkCheckMenuItem*) wdt ) );
}


/*#
    @method new_with_mnemonic GtkCheckMenuItem
    @brief Creates a new GtkCheckMenuItem containing a label.
    @param label the text of the button, with an underscore in front of the mnemonic character
    @return a new GtkCheckMenuItem

    The label will be created using gtk_label_new_with_mnemonic(), so underscores
    in label indicate the mnemonic for the menu item.
 */
FALCON_FUNC CheckMenuItem::new_with_mnemonic( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* label = args.getCString( 0 );
    GtkWidget* wdt = gtk_check_menu_item_new_with_mnemonic( label );
    vm->retval( new Gtk::CheckMenuItem( vm->findWKI( "GtkCheckMenuItem" )->asClass(),
                                        (GtkCheckMenuItem*) wdt ) );
}


//FALCON_FUNC CheckMenuItem::set_state( VMARG );


/*#
    @method get_active GtkCheckMenuItem
    @brief Returns whether the check menu item is active.
    @return TRUE if the menu item is checked.
 */
FALCON_FUNC CheckMenuItem::get_active( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_check_menu_item_get_active( (GtkCheckMenuItem*)_obj ) );
}


/*#
    @method set_active GtkCheckMenuItem
    @brief Sets the active state of the menu item's check box.
    @param is_active boolean value indicating whether the check box is active.
 */
FALCON_FUNC CheckMenuItem::set_active( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_check_menu_item_set_active( (GtkCheckMenuItem*)_obj,
                                    i_bool->asBoolean() ? TRUE : FALSE );
}


//FALCON_FUNC CheckMenuItem::set_show_toggle( VMARG );


/*#
    @method toggled GtkCheckMenuItem
    @brief Emits the GtkCheckMenuItem::toggled signal.
 */
FALCON_FUNC CheckMenuItem::toggled( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_check_menu_item_toggled( (GtkCheckMenuItem*)_obj );
}


/*#
    @method get_inconsistent GtkCheckMenuItem
    @brief Retrieves the value set by gtk_check_menu_item_set_inconsistent().
    @return TRUE if inconsistent
 */
FALCON_FUNC CheckMenuItem::get_inconsistent( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_check_menu_item_get_inconsistent( (GtkCheckMenuItem*)_obj ) );
}


/*#
    @method set_inconsistent GtkCheckMenuItem
    @brief Sets the CheckMenuItem inconsistent.
    @param setting TRUE to display an "inconsistent" third state check

    If the user has selected a range of elements (such as some text or spreadsheet
    cells) that are affected by a boolean setting, and the current values in that
    range are inconsistent, you may want to display the check in an "in between"
    state. This function turns on "in between" display. Normally you would turn
    off the inconsistent state again if the user explicitly selects a setting.
    This has to be done manually, gtk_check_menu_item_set_inconsistent() only
    affects visual appearance, it doesn't affect the semantics of the widget.
 */
FALCON_FUNC CheckMenuItem::set_inconsistent( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_check_menu_item_set_inconsistent( (GtkCheckMenuItem*)_obj,
                                          i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_draw_as_radio GtkCheckMenuItem
    @brief Sets whether check_menu_item is drawn like a GtkRadioMenuItem
    @param draw_as_radio whether check_menu_item is drawn like a GtkRadioMenuItem
 */
FALCON_FUNC CheckMenuItem::set_draw_as_radio( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_check_menu_item_set_draw_as_radio( (GtkCheckMenuItem*)_obj,
                                           i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_draw_as_radio GtkCheckMenuItem
    @brief Returns whether check_menu_item looks like a GtkRadioMenuItem
    @return Whether check_menu_item looks like a GtkRadioMenuItem
 */
FALCON_FUNC CheckMenuItem::get_draw_as_radio( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_check_menu_item_get_draw_as_radio( (GtkCheckMenuItem*)_obj ) );
}


} // Gtk
} // Falcon
