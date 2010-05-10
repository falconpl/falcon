/**
 *  \file gtk_MenuItem.cpp
 */

#include "gtk_MenuItem.hpp"

#include "gtk_Activatable.hpp"
#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void MenuItem::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_MenuItem = mod->addClass( "GtkMenuItem", &MenuItem::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkItem" ) );
    c_MenuItem->getClassDef()->addInheritance( in );

    c_MenuItem->setWKS( true );
    c_MenuItem->getClassDef()->factory( &MenuItem::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_activate",        &MenuItem::signal_activate },
    { "signal_activate_item",   &MenuItem::signal_activate_item },
    //{ "signal_toggle_size_allocate",&MenuItem::signal_toggle_size_allocate },
    //{ "signal_toggle_size_request",&MenuItem::signal_toggle_size_request },
    { "new_with_label",         &MenuItem::new_with_label },
    { "new_with_mnemonic",      &MenuItem::new_with_mnemonic },
    { "set_right_justified",    &MenuItem::set_right_justified },
    { "get_right_justified",    &MenuItem::get_right_justified },
#if GTK_MINOR_VERSION >= 16
    { "get_label",              &MenuItem::get_label },
    { "set_label",              &MenuItem::set_label },
    { "get_use_underline",      &MenuItem::get_use_underline },
    { "set_use_underline",      &MenuItem::set_use_underline },
#endif
    { "set_submenu",            &MenuItem::set_submenu },
    { "get_submenu",            &MenuItem::get_submenu },
    //{ "remove_submenu",       &MenuItem::remove_submenu },
    { "set_accel_path",         &MenuItem::set_accel_path },
    { "get_accel_path",         &MenuItem::get_accel_path },
    { "select",                 &MenuItem::select },
    { "deselect",               &MenuItem::deselect },
    { "activate",               &MenuItem::activate },
#if 0
    { "toggle_size_request",    &MenuItem::toggle_size_request },
    { "toggle_size_allocate",   &MenuItem::toggle_size_allocate },
    { "right_justify",          &MenuItem::right_justify },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_MenuItem, meth->name, meth->cb );

    Gtk::Activatable::clsInit( mod, c_MenuItem );
}


MenuItem::MenuItem( const Falcon::CoreClass* gen, const GtkMenuItem* menu )
    :
    Gtk::CoreGObject( gen, (GObject*) menu )
{}


Falcon::CoreObject* MenuItem::factory( const Falcon::CoreClass* gen, void* menu, bool )
{
    return new MenuItem( gen, (GtkMenuItem*) menu );
}


/*#
    @class GtkMenuItem
    @brief The widget used for item in menus

    The GtkMenuItem widget and the derived widgets are the only valid childs for
    menus. Their function is to correctly handle highlighting, alignment, events
    and submenus.

    As it derives from GtkBin it can hold any valid child widget, altough only a
    few are really useful.

    [...]
 */
FALCON_FUNC MenuItem::init( VMARG )
{
    MYSELF;
    if ( self->getGObject() )
        return;
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    self->setGObject( (GObject*) gtk_menu_item_new() );
}


/*#
    @method signal_activate GtkMenuItem
    @brief Emitted when the item is activated.
 */
FALCON_FUNC MenuItem::signal_activate( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "activate", (void*) &MenuItem::on_activate, vm );
}


void MenuItem::on_activate( GtkMenuItem* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "activate", "on_activate", (VMachine*)_vm );
}


/*#
    @method signal_activate_item GtkMenuItem
    @brief Emitted when the item is activated, but also if the menu item has a submenu.

    For normal applications, the relevant signal is "activate".
 */
FALCON_FUNC MenuItem::signal_activate_item( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "activate_item", (void*) &MenuItem::on_activate_item, vm );
}


void MenuItem::on_activate_item( GtkMenuItem* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "activate_item", "on_activate_item", (VMachine*)_vm );
}


#if 0
FALCON_FUNC MenuItem::signal_toggle_size_allocate( VMARG );
FALCON_FUNC MenuItem::signal_toggle_size_request( VMARG );
#endif


/*#
    @method new_with_label GtkMenuItem
    @brief Creates a new GtkMenuItem whose child is a GtkLabel.
    @param label the text for the label
    @return the newly created GtkMenuItem
 */
FALCON_FUNC MenuItem::new_with_label( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* label = args.getCString( 0 );
    GtkWidget* wdt = gtk_menu_item_new_with_label( label );
    vm->retval( new Gtk::MenuItem(
                vm->findWKI( "GtkMenuItem" )->asClass(), (GtkMenuItem*) wdt ) );
}


/*#
    @method new_with_mnemonic GtkMenuItem
    @brief Creates a new GtkMenuItem containing a label.
    @param label The text of the button, with an underscore in front of the mnemonic character
    @return a new GtkMenuItem

    The label will be created using gtk_label_new_with_mnemonic(), so underscores
    in label indicate the mnemonic for the menu item.
 */
FALCON_FUNC MenuItem::new_with_mnemonic( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* label = args.getCString( 0 );
    GtkWidget* wdt = gtk_menu_item_new_with_mnemonic( label );
    vm->retval( new Gtk::MenuItem(
                vm->findWKI( "GtkMenuItem" )->asClass(), (GtkMenuItem*) wdt ) );
}


/*#
    @method set_right_justified GtkMenuItem
    @brief Sets whether the menu item appears justified at the right side of a menu bar.
    @param right_justified if TRUE the menu item will appear at the far right if added to a menu bar.

    This was traditionally done for "Help" menu items, but is now considered
    a bad idea. (If the widget layout is reversed for a right-to-left language
    like Hebrew or Arabic, right-justified-menu-items appear at the left.)
 */
FALCON_FUNC MenuItem::set_right_justified( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_item_set_right_justified( (GtkMenuItem*)_obj,
                                       i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_right_justified GtkMenuItem
    @brief Gets whether the menu item appears justified at the right side of the menu bar.
    @return TRUE if the menu item will appear at the far right if added to a menu bar.
 */
FALCON_FUNC MenuItem::get_right_justified( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_menu_item_get_right_justified( (GtkMenuItem*)_obj ) );
}


#if GTK_MINOR_VERSION >= 16
/*#
    @method get_label GtkMenuItem
    @brief Gets the text on the menu_item label
    @return The text in the menu_item label.
 */
FALCON_FUNC MenuItem::get_label( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( UTF8String( gtk_menu_item_get_label( (GtkMenuItem*)_obj ) ) );
}


/*#
    @method set_label GtkMenuItem
    @brief Sets text on the menu_item label
    @param label the text you want to set
 */
FALCON_FUNC MenuItem::set_label( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* label = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_menu_item_set_label( (GtkMenuItem*)_obj, label );
}


/*#
    @method get_use_underline GtkMenuItem
    @brief Checks if an underline in the text indicates the next character should be used for the mnemonic accelerator key.
    @return TRUE if an embedded underline in the label indicates the mnemonic accelerator key.
 */
FALCON_FUNC MenuItem::get_use_underline( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_menu_item_get_use_underline( (GtkMenuItem*)_obj ) );
}


/*#
    @method set_use_underline GtkMenuItem
    @brief If true, an underline in the text indicates the next character should be used for the mnemonic accelerator key.
    @param setting TRUE if underlines in the text indicate mnemonics
 */
FALCON_FUNC MenuItem::set_use_underline( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_item_set_use_underline( (GtkMenuItem*)_obj,
                                     i_bool->asBoolean() ? TRUE : FALSE );
}
#endif // GTK_MINOR_VERSION >= 16


/*#
    @method set_submenu GtkMenuItem
    @brief Sets or replaces the menu item's submenu, or removes it when a NULL submenu is passed.
    @param submenu the submenu, or NULL.
 */
FALCON_FUNC MenuItem::set_submenu( VMARG )
{
    Item* i_menu = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_menu || !( i_menu->isNil() || ( i_menu->isObject()
        && IS_DERIVED( i_menu, GtkWidget ) ) ) )
        throw_inv_params( "[GtkWidget]" );
#endif
    GtkWidget* menu = i_menu->isNil() ? NULL :
                            (GtkWidget*) COREGOBJECT( i_menu )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_item_set_submenu( (GtkMenuItem*)_obj, menu );
}


/*#
    @method get_submenu GtkMenuItem
    @brief Gets the submenu underneath this menu item, if any.
    @return submenu for this menu item, or NULL if none.
 */
FALCON_FUNC MenuItem::get_submenu( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* menu = gtk_menu_item_get_submenu( (GtkMenuItem*)_obj );
    if ( menu )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), menu ) );
    else
        vm->retnil();
}


//FALCON_FUNC MenuItem::remove_submenu( VMARG );


/*#
    @method set_accel_path GtkMenuItem
    @brief Set the accelerator path on menu_item, through which runtime changes of the menu item's accelerator caused by the user can be identified and saved to persistant storage (see gtk_accel_map_save() on this).
    @param accel_path accelerator path, corresponding to this menu item's functionality, or NULL to unset the current path

    To setup a default accelerator for this menu item, call gtk_accel_map_add_entry()
    with the same accel_path. See also gtk_accel_map_add_entry() on the specifics
    of accelerator paths, and gtk_menu_set_accel_path() for a more convenient
    variant of this function.

    This function is basically a convenience wrapper that handles calling
    gtk_widget_set_accel_path() with the appropriate accelerator group for the menu item.

    Note that you do need to set an accelerator on the parent menu with
    gtk_menu_set_accel_group() for this to work.
 */
FALCON_FUNC MenuItem::set_accel_path( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* path = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_menu_item_set_accel_path( (GtkMenuItem*)_obj, path );
}


/*#
    @method get_accel_path GtkMenuItem
    @brief Retrieve the accelerator path that was previously set on menu_item.
    @return the accelerator path corresponding to this menu item's functionality, or NULL if not set
 */
FALCON_FUNC MenuItem::get_accel_path( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* path = gtk_menu_item_get_accel_path( (GtkMenuItem*)_obj );
    if ( path )
        vm->retval( UTF8String( path ) );
    else
        vm->retnil();
}


/*#
    @method select GtkMenuItem
    @brief Emits the "select" signal on the given item.

    Behaves exactly like gtk_item_select.
 */
FALCON_FUNC MenuItem::select( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_item_select( (GtkMenuItem*)_obj );
}


/*#
    @method deselect GtkMenuItem
    @brief Emits the "deselect" signal on the given item.

    Behaves exactly like gtk_item_deselect.
 */
FALCON_FUNC MenuItem::deselect( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_item_deselect( (GtkMenuItem*)_obj );
}


/*#
    @method activate GtkMenuItem
    @brief Emits the "activate" signal on the given item.
 */
FALCON_FUNC MenuItem::activate( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_menu_item_activate( (GtkMenuItem*)_obj );
}


#if 0
FALCON_FUNC MenuItem::toggle_size_request( VMARG )
FALCON_FUNC MenuItem::toggle_size_allocate( VMARG );
FALCON_FUNC MenuItem::right_justify( VMARG );
#endif


} // Gtk
} // Falcon
