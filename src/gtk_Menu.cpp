/**
 *  \file gtk_Menu.cpp
 */

#include "gtk_Menu.hpp"

#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Menu::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Menu = mod->addClass( "GtkMenu", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkMenuShell" ) );
    c_Menu->getClassDef()->addInheritance( in );

    c_Menu->setWKS( true );
    c_Menu->getClassDef()->factory( &Menu::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_screen",         &Menu::set_screen },
#if 0 // deprecated
    { "append",             &Menu::append },
    { "prepend",            &Menu::prepend },
    { "insert",             &Menu::insert },
#endif
    { "reorder_child",      &Menu::reorder_child },
    { "attach",             &Menu::attach },
    { "popup",              &Menu::popup },
#if 0
    { "set_accel_group",    &Menu::set_accel_group },
    { "get_accel_group",    &Menu::get_accel_group },
#endif
    { "set_accel_path",     &Menu::set_accel_path },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_accel_path",     &Menu::get_accel_path },
#endif
    { "set_title",          &Menu::set_title },
    { "get_title",          &Menu::get_title },
    { "set_monitor",        &Menu::set_monitor },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_monitor",        &Menu::get_monitor },
#endif
    { "get_tearoff_state",  &Menu::get_tearoff_state },
#if GTK_CHECK_VERSION( 2, 18, 0 )
    { "set_reserve_toggle_size",&Menu::set_reserve_toggle_size },
    { "get_reserve_toggle_size",&Menu::get_reserve_toggle_size },
#endif
    { "popdown",            &Menu::popdown },
    { "reposition",         &Menu::reposition },
    { "get_active",         &Menu::get_active },
    { "set_active",         &Menu::set_active },
    { "set_tearoff_state",  &Menu::set_tearoff_state },
    { "attach_to_widget",   &Menu::attach_to_widget },
    { "detach",             &Menu::detach },
    { "get_attach_widget",  &Menu::get_attach_widget },
    { "get_for_attach_widget",&Menu::get_for_attach_widget },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Menu, meth->name, meth->cb );
}


Menu::Menu( const Falcon::CoreClass* gen, const GtkMenu* menu )
    :
    Gtk::CoreGObject( gen, (GObject*) menu )
{}


Falcon::CoreObject* Menu::factory( const Falcon::CoreClass* gen, void* menu, bool )
{
    return new Menu( gen, (GtkMenu*) menu );
}


/*#
    @class GtkMenu
    @brief A menu widget

    A GtkMenu is a GtkMenuShell that implements a drop down menu consisting of a
    list of GtkMenuItem objects which can be navigated and activated by the user
    to perform application functions.

    A GtkMenu is most commonly dropped down by activating a GtkMenuItem in a
    GtkMenuBar or popped up by activating a GtkMenuItem in another GtkMenu.

    A GtkMenu can also be popped up by activating a GtkOptionMenu. Other composite
    widgets such as the GtkNotebook can pop up a GtkMenu as well.

    Applications can display a GtkMenu as a popup menu by calling the gtk_menu_popup()
    function. The example below shows how an application can pop up a menu when
    the 3rd mouse button is pressed.

    [...]
 */
FALCON_FUNC Menu::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_menu_new() );
}


/*#
    @method signal_move_scroll GtkMenu
    @brief .
 */
FALCON_FUNC Menu::signal_move_scroll( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "move_scroll", (void*) &Menu::on_move_scroll, vm );
}


void Menu::on_move_scroll( GtkMenu* obj, GtkScrollType scroll_type, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "move_scroll", false );

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
                || !it.asObject()->getMethod( "on_move_scroll", it ) )
            {
                printf(
                "[GtkMenu::on_move_scroll] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) scroll_type );
        vm->callItem( it, 1 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method set_screen GtkMenu
    @brief Sets the GdkScreen on which the menu will be displayed.
    @param a GdkScreen, or NULL if the screen should be determined by the widget the menu is attached to
 */
FALCON_FUNC Menu::set_screen( VMARG )
{
    Item* i_screen = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_screen && !( i_screen->isNil()
        || ( i_screen->isObject() && IS_DERIVED( i_screen, GdkScreen ) ) ) )
        throw_inv_params( "[GdkScreen]" );
#endif
    GdkScreen* screen = i_screen && i_screen->isNil() ? NULL
            : (GdkScreen*) COREGOBJECT( i_screen )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_set_screen( (GtkMenu*)_obj, screen );
}


#if 0 // deprecated
FALCON_FUNC Menu::append( VMARG );
FALCON_FUNC Menu::prepend( VMARG );
FALCON_FUNC Menu::insert( VMARG );
#endif


/*#
    @method reorder_child GtkMenu
    @brief Moves a GtkMenuItem to a new position within the GtkMenu.
    @param child the GtkMenuItem to move.
    @param position the new position to place child. Positions are numbered from 0 to n-1.
 */
FALCON_FUNC Menu::reorder_child( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || i_child->isNil() || !i_child->isObject()
        || !IS_DERIVED( i_child, GtkMenuItem )
        || !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "GtkMenuItem,I" );
#endif
    GtkWidget* child = (GtkWidget*) COREGOBJECT( i_child )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_reorder_child( (GtkMenu*)_obj, child, i_pos->asInteger() );
}


/*#
    @method attach GtkMenu
    @brief Adds a new GtkMenuItem to a (table) menu.
    @param child a GtkMenuItem.
    @param left_attach The column number to attach the left side of the item to.
    @param right_attach The column number to attach the right side of the item to.
    @param top_attach The row number to attach the top of the item to.
    @param bottom_attach The row number to attach the bottom of the item to.

    The number of 'cells' that an item will occupy is specified by left_attach,
    right_attach, top_attach and bottom_attach. These each represent the leftmost,
    rightmost, uppermost and lower column and row numbers of the table.
    (Columns and rows are indexed from zero).

    Note that this function is not related to gtk_menu_detach().
 */
FALCON_FUNC Menu::attach( VMARG )
{
    const char* spec = "GtkMenuItem,I,I,I,I";
    Gtk::ArgCheck0 args( vm, spec );
    CoreGObject* o_child = args.getCoreGObject( 0 );
    guint left_a = args.getInteger( 1 );
    guint right_a = args.getInteger( 2 );
    guint top_a = args.getInteger( 3 );
    guint bottom_a = args.getInteger( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_child, GtkMenuItem ) )
        throw_inv_params( spec );
#endif
    GtkWidget* child = (GtkWidget*) o_child->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_attach( (GtkMenu*)_obj, child, left_a, right_a, top_a, bottom_a );
}


/*#
    @method popup GtkMenu
    @brief Displays a menu and makes it available for selection.
    @param parent_menu_shell the menu shell containing the triggering menu item, or NULL.
    @param parent_menu_item the menu item whose activation triggered the popup, or NULL.
    @param button the mouse button which was pressed to initiate the event.
    @param activate_time the time at which the activation event occurred.

    Applications can use this function to display context-sensitive menus, and
    will typically supply NULL for the parent_menu_shell, parent_menu_item, func
    and data parameters. The default menu positioning function will position
    the menu at the current mouse cursor position.

    [...]
 */
FALCON_FUNC Menu::popup( VMARG )
{ // missing func and data params
    const char* spec = "GtkWidget,GtkWidget,I,I";
    Gtk::ArgCheck0 args( vm, spec );
    CoreGObject* o_mshell = args.getCoreGObject( 0, false );
    CoreGObject* o_mitem = args.getCoreGObject( 1, false );
    guint btn = args.getInteger( 2 );
    guint32 time = args.getInteger( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( ( o_mshell && !CoreObject_IS_DERIVED( o_mshell, GtkWidget ) )
        || ( o_mitem && !CoreObject_IS_DERIVED( o_mitem, GtkWidget ) ) )
        throw_inv_params( spec );
#endif
    GtkWidget* mshell = o_mshell ? (GtkWidget*) o_mshell->getObject() : NULL;
    GtkWidget* mitem = o_mitem ? (GtkWidget*) o_mitem->getObject() : NULL;
    MYSELF;
    GET_OBJ( self );
    gtk_menu_popup( (GtkMenu*)_obj, mshell, mitem, NULL, NULL, btn, time );
}

#if 0
/*#
    @method set_accel_group GtkMenu
    @brief Set the GtkAccelGroup which holds global accelerators for the menu.
    @param accel_group a GtkAccelGroup, or nil.

    This accelerator group needs to also be added to all windows that this menu
    is being used in with gtk_window_add_accel_group(), in order for those windows
    to support all the accelerators contained in this group.
 */
FALCON_FUNC Menu::set_accel_group( VMARG )
{
    Item* i_grp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_grp && !( i_grp->isNil()
        || ( i_grp->isObject() && IS_DERIVED( i_grp, GtkAccelGroup ) ) ) )
        throw_inv_params( "[GtkAccelGroup]" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_set_accel_group( (GtkMenu*)_obj,
            i_grp ? (GtkAccelGroup*) COREGOBJECT( i_grp )->getObject() : NULL );
}


FALCON_FUNC Menu::get_accel_group( VMARG );
#endif

/*#
    @method set_accel_path GtkMenu
    @brief Sets an accelerator path for this menu from which accelerator paths for its immediate children, its menu items, can be constructed.
    @param accel_path a valid accelerator path, or nil.
    [...]
 */
FALCON_FUNC Menu::set_accel_path( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const gchar* accel = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_menu_set_accel_path( (GtkMenu*)_obj, accel );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_accel_path GtkMenu
    @brief Retrieves the accelerator path set on the menu.
    @return the accelerator path set on the menu.
 */
FALCON_FUNC Menu::get_accel_path( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* s = gtk_menu_get_accel_path( (GtkMenu*)_obj );
    if ( s )
        vm->retval( UTF8String( s ) );
    else
        vm->retnil();
}
#endif


/*#
    @method set_title GtkMenu
    @brief Sets the title string for the menu.
    @param title a string containing the title for the menu , or nil.

    The title is displayed when the menu is shown as a tearoff menu. If title
    is NULL, the menu will see if it is attached to a parent menu item, and
    if so it will try to use the same text as that menu item's label.
 */
FALCON_FUNC Menu::set_title( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const gchar* title = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_menu_set_title( (GtkMenu*)_obj, title );
}


/*#
    @method get_title GtkMenu
    @brief Returns the title of the menu.
    @return the title of the menu, or nil if the menu has no title set on it.
 */
FALCON_FUNC Menu::get_title( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* title = gtk_menu_get_title( (GtkMenu*)_obj );
    if ( title )
        vm->retval( UTF8String( title ) );
    else
        vm->retnil();
}


/*#
    @method set_monitor GtkMenu
    @brief Informs GTK+ on which monitor a menu should be popped up.
    @param monitor_num the number of the monitor on which the menu should be popped up

    [...]
 */
FALCON_FUNC Menu::set_monitor( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || i_num->isNil() || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_set_monitor( (GtkMenu*)_obj, i_num->asInteger() );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_monitor GtkMenu
    @brief Retrieves the number of the monitor on which to show the menu.
    @return the number of the monitor on which the menu should be popped up or -1, if no monitor has been set
 */
FALCON_FUNC Menu::get_monitor( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_menu_get_monitor( (GtkMenu*)_obj ) );
}
#endif


/*#
    @method get_tearoff_state GtkMenu
    @brief Returns whether the menu is torn off.
    @return true if the menu is currently torn off.
 */
FALCON_FUNC Menu::get_tearoff_state( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_menu_get_tearoff_state( (GtkMenu*)_obj ) );
}


#if GTK_CHECK_VERSION( 2, 18, 0 )
/*#
    @method set_reserve_toggle_size GtkMenu
    @brief Sets whether the menu should reserve space for drawing toggles or icons, regardless of their actual presence.
    @param reserve_toggle_size whether to reserve size for toggles
 */
FALCON_FUNC Menu::set_reserve_toggle_size( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_set_reserve_toggle_size( (GtkMenu*)_obj,
                                      i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_reserve_toggle_size GtkMenu
    @brief Returns whether the menu reserves space for toggles and icons, regardless of their actual presence.
    @return Whether the menu reserves toggle space
 */
FALCON_FUNC Menu::get_reserve_toggle_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_menu_get_reserve_toggle_size( (GtkMenu*)_obj ) );
}
#endif


/*#
    @method popdown GtkMenu
    @brief Removes the menu from the screen.
 */
FALCON_FUNC Menu::popdown( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_menu_popdown( (GtkMenu*)_obj );
}


/*#
    @method reposition GtkMenu
    @brief Repositions the menu according to its position function.
 */
FALCON_FUNC Menu::reposition( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_menu_reposition( (GtkMenu*)_obj );
}


/*#
    @method get_active GtkMenu
    @brief Returns the selected menu item from the menu.
    @return the GtkMenuItem that was last selected in the menu. If a selection has not yet been made, the first menu item is selected.
 */
FALCON_FUNC Menu::get_active( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_menu_get_active( (GtkMenu*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}


/*#
    @method set_active GtkMenu
    @brief Selects the specified menu item within the menu.
    @param index_ the index of the menu item to select. Index values are from 0 to n-1.
 */
FALCON_FUNC Menu::set_active( VMARG )
{
    Item* i_ind = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ind || i_ind->isNil() || !i_ind->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_set_active( (GtkMenu*)_obj, i_ind->asInteger() );
}


/*#
    @method set_tearoff_state GtkMenu
    @brief Changes the tearoff state of the menu.
    @param torn_off If true, menu is displayed as a tearoff menu.

    A menu is normally displayed as drop down menu which persists as long as the
    menu is active. It can also be displayed as a tearoff menu which persists
    until it is closed or reattached.
 */
FALCON_FUNC Menu::set_tearoff_state( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_set_tearoff_state( (GtkMenu*)_obj,
                                i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method attach_to_widget GtkMenu
    @brief Attaches the menu to the widget and provides a callback function that will be invoked when the menu calls gtk_menu_detach() during its destruction.
    @param attach_widget the GtkWidget that the menu will be attached to.

 */
FALCON_FUNC Menu::attach_to_widget( VMARG )
{ // missing detacher param
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_attach_to_widget( (GtkMenu*)_obj, wdt, NULL );
}


/*#
    @method detach GtkMenu
    @brief Detaches the menu from the widget to which it had been attached.

    This function will call the callback function, detacher, provided when the
    gtk_menu_attach_to_widget() function was called.
 */
FALCON_FUNC Menu::detach( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_menu_detach( (GtkMenu*)_obj );
}


/*#
    @method get_attach_widget GtkMenu
    @brief Returns the GtkWidget that the menu is attached to.
    @return the GtkWidget that the menu is attached to.
 */
FALCON_FUNC Menu::get_attach_widget( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_menu_get_attach_widget( (GtkMenu*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


/*#
    @method get_for_attach_widget GtkMenu
    @brief Returns a list of the menus which are attached to this widget.
    @param widget a GtkWidget
    @return the list of menus attached to his widget
 */
FALCON_FUNC Menu::get_for_attach_widget( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    GList* lst = gtk_menu_get_for_attach_widget( (GtkWidget*) wdt );
    GList* el;
    int num = 0;
    for ( el = lst; el; el = el->next, ++num );
    CoreArray* arr = new CoreArray( num );
    if ( num )
    {
        Item* wki = vm->findWKI( "GtkMenu" );
        for ( el = lst; el; el = el->next )
            arr->append( new Gtk::Menu( wki->asClass(), (GtkMenu*) el->data ) );
    }
    vm->retval( arr );
}


} // Gtk
} // Falcon
