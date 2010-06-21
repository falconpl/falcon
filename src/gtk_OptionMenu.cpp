/**
 *  \file gtk_OptionMenu.cpp
 */

#include "gtk_OptionMenu.hpp"

#include "gtk_Menu.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void OptionMenu::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_OptionMenu = mod->addClass( "GtkOptionMenu", &OptionMenu::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkButton" ) );
    c_OptionMenu->getClassDef()->addInheritance( in );

    //c_OptionMenu->setWKS( true );
    c_OptionMenu->getClassDef()->factory( &OptionMenu::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_changed", &OptionMenu::signal_changed },
    { "get_menu",       &OptionMenu::get_menu },
    { "set_menu",       &OptionMenu::set_menu },
    { "remove_menu",    &OptionMenu::remove_menu },
    { "set_history",    &OptionMenu::set_history },
    { "get_history",    &OptionMenu::get_history },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_OptionMenu, meth->name, meth->cb );
}


OptionMenu::OptionMenu( const Falcon::CoreClass* gen, const GtkOptionMenu* menu )
    :
    Gtk::CoreGObject( gen, (GObject*) menu )
{}


Falcon::CoreObject* OptionMenu::factory( const Falcon::CoreClass* gen, void* menu, bool )
{
    return new OptionMenu( gen, (GtkOptionMenu*) menu );
}


/*#
    @class GtkOptionMenu
    @brief A widget used to choose from a list of valid choices

    A GtkOptionMenu is a widget that allows the user to choose from a list of
    valid choices. The GtkOptionMenu displays the selected choice.
    When activated the GtkOptionMenu displays a popup GtkMenu which allows
    the user to make a new choice.

    Using a GtkOptionMenu is simple; build a GtkMenu, by calling gtk_menu_new(),
    then appending menu items to it with gtk_menu_shell_append(). Set that menu
    on the option menu with gtk_option_menu_set_menu(). Set the selected menu
    item with gtk_option_menu_set_history(); connect to the "changed" signal on
    the option menu; in the "changed" signal, check the new selected menu item
    with gtk_option_menu_get_history().

    As of GTK+ 2.4, GtkOptionMenu has been deprecated in favor of GtkComboBox.
 */
FALCON_FUNC OptionMenu::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_option_menu_new() );
}


/*#
    @method signal_changed GtkOptionMenu
    @brief .
 */
FALCON_FUNC OptionMenu::signal_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "changed", (void*) &OptionMenu::on_changed, vm );
}


void OptionMenu::on_changed( GtkOptionMenu* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "changed", "on_changed", (VMachine*)_vm );
}


/*#
    @method get_menu GtkOptionMenu
    @brief Returns the GtkMenu associated with the GtkOptionMenu.
    @return the GtkMenu associated with the GtkOptionMenu.
 */
FALCON_FUNC OptionMenu::get_menu( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* menu = gtk_option_menu_get_menu( (GtkOptionMenu*)_obj );
    vm->retval( new Gtk::Menu( vm->findWKI( "GtkMenu" )->asClass(), (GtkMenu*) menu ) );
}


/*#
    @method set_menu GtkOptionMenu
    @brief Provides the GtkMenu that is popped up to allow the user to choose a new value.
    @param menu the GtkMenu to associate with the GtkOptionMenu.

    You should provide a simple menu avoiding the use of tearoff menu items,
    submenus, and accelerators.
 */
FALCON_FUNC OptionMenu::set_menu( VMARG )
{
    Item* i_menu = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_menu || !i_menu->isObject() || !IS_DERIVED( i_menu, GtkMenu ) )
        throw_inv_params( "GtkMenu" );
#endif
    GtkWidget* menu = (GtkWidget*) COREGOBJECT( i_menu )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_option_menu_set_menu( (GtkOptionMenu*)_obj, menu );
}


/*#
    @method remove_menu GtkOptionMenu
    @brief Removes the menu from the option menu.
 */
FALCON_FUNC OptionMenu::remove_menu( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_option_menu_remove_menu( (GtkOptionMenu*)_obj );
}


/*#
    @method set_history GtkOptionMenu
    @brief Selects the menu item specified by index_ making it the newly selected value for the option menu.
    @param index the index of the menu item to select. Index values are from 0 to n-1.
 */
FALCON_FUNC OptionMenu::set_history( VMARG )
{
    Item* i_idx = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_idx || !i_idx->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_option_menu_set_history( (GtkOptionMenu*)_obj, i_idx->asInteger() );
}


/*#
    @method get_history GtkOptionMenu
    @brief Retrieves the index of the currently selected menu item.
    @return index of the selected menu item, or -1 if there are no menu items

    The menu items are numbered from top to bottom, starting with 0.
 */
FALCON_FUNC OptionMenu::get_history( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_option_menu_get_history( (GtkOptionMenu*)_obj ) );
}


} // Gtk
} // Falcon
