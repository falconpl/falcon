/**
 *  \file gtk_MenuShell.cpp
 */

#include "gtk_MenuShell.hpp"

#include "gtk_Buildable.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void MenuShell::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_MenuShell = mod->addClass( "GtkMenuShell", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_MenuShell->getClassDef()->addInheritance( in );

    c_MenuShell->getClassDef()->factory( &MenuShell::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_activate_current",&MenuShell::signal_activate_current },
    { "signal_deactivate",      &MenuShell::signal_deactivate },
    { "signal_cycle_focus",     &MenuShell::signal_cycle_focus },
    { "signal_deactivate",      &MenuShell::signal_deactivate },
    { "signal_move_current",    &MenuShell::signal_move_current },
    { "signal_move_selected",   &MenuShell::signal_move_selected },
    { "signal_selection_done",  &MenuShell::signal_selection_done },
    { "append",         &MenuShell::append },
    { "prepend",        &MenuShell::prepend },
    { "insert",         &MenuShell::insert },
    { "deactivate",     &MenuShell::deactivate },
    { "select_item",    &MenuShell::select_item },
    { "select_first",   &MenuShell::select_first },
    { "deselect",       &MenuShell::deselect },
    { "activate_item",  &MenuShell::activate_item },
    { "deactivate",     &MenuShell::deactivate },
    { "set_take_focus", &MenuShell::set_take_focus },
    { "get_take_focus", &MenuShell::get_take_focus },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_MenuShell, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_MenuShell );
}


MenuShell::MenuShell( const Falcon::CoreClass* gen, const GtkMenuShell* shell )
    :
    Gtk::CoreGObject( gen, (GObject*) shell )
{}


Falcon::CoreObject* MenuShell::factory( const Falcon::CoreClass* gen, void* shell, bool )
{
    return new MenuShell( gen, (GtkMenuShell*) shell );
}


/*#
    @class GtkMenuShell
    @brief A base class for menu objects.
 */


/*#
    @method signal_activate_current GtkMenuShell
    @brief An action signal that activates the current menu item within the menu shell.
 */
FALCON_FUNC MenuShell::signal_activate_current( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "activate_current", (void*) &MenuShell::on_activate_current, vm );
}


void MenuShell::on_activate_current( GtkMenuShell* obj, gboolean force_hide, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "activate_current", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*)_vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();
        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_activate_current", it ) )
            {
                printf(
                "[GtkMenuShell::on_activate_current] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (bool) force_hide );
        vm->callItem( it, 1 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_cancel GtkMenuShell
    @brief An action signal which cancels the selection within the menu shell.

    Causes the GtkMenuShell::selection-done signal to be emitted.
 */
FALCON_FUNC MenuShell::signal_cancel( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "cancel", (void*) &MenuShell::on_cancel, vm );
}


void MenuShell::on_cancel( GtkMenuShell* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "cancel", "on_cancel", (VMachine*)_vm );
}


/*#
    @method signal_cycle_focus GtkMenuShell
    @brief .
 */
FALCON_FUNC MenuShell::signal_cycle_focus( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "cycle_focus", (void*) &MenuShell::on_cycle_focus, vm );
}


void MenuShell::on_cycle_focus( GtkMenuShell* obj, GtkDirectionType dir, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "cycle_focus", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*)_vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();
        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_cycle_focus", it ) )
            {
                printf(
                "[GtkMenuShell::on_cycle_focus] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) dir );
        vm->callItem( it, 1 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_deactivate GtkMenuShell
    @brief This signal is emitted when a menu shell is deactivated.
 */
FALCON_FUNC MenuShell::signal_deactivate( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "deactivate", (void*) &MenuShell::on_deactivate, vm );
}


void MenuShell::on_deactivate( GtkMenuShell* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "deactivate", "on_deactivate", (VMachine*)_vm );
}


/*#
    @method signal_move_current GtkMenuShell
    @brief An action signal which moves the current menu item in the direction specified by direction.
 */
FALCON_FUNC MenuShell::signal_move_current( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "move_current", (void*) &MenuShell::on_move_current, vm );
}


void MenuShell::on_move_current( GtkMenuShell* obj, GtkMenuDirectionType dir, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "move_current", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*)_vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();
        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_move_current", it ) )
            {
                printf(
                "[GtkMenuShell::on_move_current] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) dir );
        vm->callItem( it, 1 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_move_selected GtkMenuShell
    @brief The move-selected signal is emitted to move the selection to another item.

    Return true to stop the signal emission, false to continue.
 */
FALCON_FUNC MenuShell::signal_move_selected( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "move_selected", (void*) &MenuShell::on_move_selected, vm );
}


gboolean MenuShell::on_move_selected( GtkMenuShell* obj, gint dist, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "move_selected", false );

    if ( !cs || cs->empty() )
        return FALSE; // continue

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_move_selected", it ) )
            {
                printf(
                "[GtkWidget::on_move_selected] invalid callback (expected callable)\n" );
                return TRUE; // block
            }
        }
        vm->pushParam( dist );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_move_selected] invalid callback (expected boolean)\n" );
            return TRUE; // block
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // continue
}


/*#
    @method signal_selection_done GtkMenuShell
    @brief This signal is emitted when a selection has been completed within a menu shell.
 */
FALCON_FUNC MenuShell::signal_selection_done( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "selection_done", (void*) &MenuShell::on_selection_done, vm );
}


void MenuShell::on_selection_done( GtkMenuShell* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "selection_done", "on_selection_done", (VMachine*)_vm );
}


/*#
    @method append GtkMenuShell
    @brief Adds a new GtkMenuItem to the end of the menu shell's item list.
    @param child The GtkMenuItem to add.
 */
FALCON_FUNC MenuShell::append( VMARG )
{
    Item* i_chld = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_chld || i_chld->isNil() || !i_chld->isObject()
        || !IS_DERIVED( i_chld, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* chld = (GtkWidget*) COREGOBJECT( i_chld )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_append( (GtkMenuShell*)_obj, chld );
}


/*#
    @method prepend GtkMenuShell
    @brief Adds a new GtkMenuItem to the beginning of the menu shell's item list.
    @param child The GtkMenuItem to add.
 */
FALCON_FUNC MenuShell::prepend( VMARG )
{
    Item* i_chld = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_chld || i_chld->isNil() || !i_chld->isObject()
        || !IS_DERIVED( i_chld, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* chld = (GtkWidget*) COREGOBJECT( i_chld )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_prepend( (GtkMenuShell*)_obj, chld );
}


/*#
    @method insert GtkMenuShell
    @brief Adds a new GtkMenuItem to the menu shell's item list at the position indicated by position.
    @param child The GtkMenuItem to add.
    @param position The position in the item list where child is added. Positions are numbered from 0 to n-1.
 */
FALCON_FUNC MenuShell::insert( VMARG )
{
    Item* i_chld = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_chld || i_chld->isNil() || !i_chld->isObject()
        || !IS_DERIVED( i_chld, GtkWidget )
        || !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "GtkWidget,I" );
#endif
    GtkWidget* chld = (GtkWidget*) COREGOBJECT( i_chld )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_insert( (GtkMenuShell*)_obj, chld, i_pos->asInteger() );
}


/*#
    @method deactivate GtkMenuShell
    @brief Deactivates the menu shell. Typically this results in the menu shell being erased from the screen.
 */
FALCON_FUNC MenuShell::deactivate( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_deactivate( (GtkMenuShell*)_obj );
}


/*#
    @method select_item GtkMenuShell
    @brief Selects the menu item from the menu shell.
    @param menu_item The GtkMenuItem to select.
 */
FALCON_FUNC MenuShell::select_item( VMARG )
{
    Item* i_chld = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_chld || i_chld->isNil() || !i_chld->isObject()
        || !IS_DERIVED( i_chld, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* chld = (GtkWidget*) COREGOBJECT( i_chld )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_select_item( (GtkMenuShell*)_obj, chld );
}


/*#
    @method select_first GtkMenuShell
    @brief Select the first visible or selectable child of the menu shell; don't select tearoff items unless the only item is a tearoff item.
    @param search_sensitive if true, search for the first selectable menu item, otherwise select nothing if the first item isn't sensitive. This should be false if the menu is being popped up initially.
 */
FALCON_FUNC MenuShell::select_first( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_select_first( (GtkMenuShell*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method deselect GtkMenuShell
    @brief Deselects the currently selected item from the menu shell, if any.
 */
FALCON_FUNC MenuShell::deselect( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_deselect( (GtkMenuShell*)_obj );
}


/*#
    @method activate_item GtkMenuShell
    @brief Activates the menu item within the menu shell.
    @param menu_item The GtkMenuItem to activate
    @param force_deactivate If true, force the deactivation of the menu shell after the menu item is activated.

 */
FALCON_FUNC MenuShell::activate_item( VMARG )
{
    Item* i_chld = vm->param( 0 );
    Item* i_bool = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_chld || i_chld->isNil() || !i_chld->isObject()
        || !IS_DERIVED( i_chld, GtkWidget )
        || !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "GtkWidget,B" );
#endif
    GtkWidget* chld = (GtkWidget*) COREGOBJECT( i_chld )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_activate_item( (GtkMenuShell*)_obj, chld, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method cancel GtkMenuShell
    @brief Cancels the selection within the menu shell.
 */
FALCON_FUNC MenuShell::cancel( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_cancel( (GtkMenuShell*)_obj );
}


/*#
    @method set_take_focus GtkMenuShell
    @brief If take_focus is TRUE (the default) the menu shell will take the keyboard focus so that it will receive all keyboard events which is needed to enable keyboard navigation in menus.
    @param take_focus true if the menu shell should take the keyboard focus on popup.

    Setting take_focus to FALSE is useful only for special applications like virtual
    keyboard implementations which should not take keyboard focus.

    The take_focus state of a menu or menu bar is automatically propagated to
    submenus whenever a submenu is popped up, so you don't have to worry about
    recursively setting it for your entire menu hierarchy. Only when programmatically
    picking a submenu and popping it up manually, the take_focus property of
    the submenu needs to be set explicitely.

    Note that setting it to FALSE has side-effects:
    If the focus is in some other app, it keeps the focus and keynav in the
    menu doesn't work. Consequently, keynav on the menu will only work if the
    focus is on some toplevel owned by the onscreen keyboard.

    To avoid confusing the user, menus with take_focus set to FALSE should not
    display mnemonics or accelerators, since it cannot be guaranteed that they will work.
 */
FALCON_FUNC MenuShell::set_take_focus( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_shell_set_take_focus( (GtkMenuShell*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_take_focus GtkMenuShell
    @brief Returns true if the menu shell will take the keyboard focus on popup.
    @return true if the menu shell will take the keyboard focus on popup.
 */
FALCON_FUNC MenuShell::get_take_focus( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_menu_shell_get_take_focus( (GtkMenuShell*)_obj ) );
}


} // Gtk
} // Falcon
