/**
 *  \file gtk_Action.cpp
 */

#include "gtk_Action.hpp"

#include "gtk_Buildable.hpp"
#include "gtk_Image.hpp"
#include "gtk_Menu.hpp"
#include "gtk_MenuItem.hpp"
#include "gtk_ToolItem.hpp"
#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Action::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Action = mod->addClass( "GtkAction", &Action::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_Action->getClassDef()->addInheritance( in );

    c_Action->setWKS( true );
    c_Action->getClassDef()->factory( &Action::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_activate",        &Action::signal_activate },
    { "get_name",               &Action::get_name },
    { "is_sensitive",           &Action::is_sensitive },
    { "get_sensitive",          &Action::get_sensitive },
    { "set_sensitive",          &Action::set_sensitive },
    { "is_visible",             &Action::is_visible },
    { "get_visible",            &Action::get_visible },
    { "set_visible",            &Action::set_visible },
    { "activate",               &Action::activate },
    { "create_icon",            &Action::create_icon },
    { "create_menu_item",       &Action::create_menu_item },
    { "create_tool_item",       &Action::create_tool_item },
    { "create_menu",            &Action::create_menu },
#if 0 // deprecated
    { "connect_proxy",          &Action::connect_proxy },
    { "disconnect_proxy",       &Action::disconnect_proxy },
#endif
    { "get_proxies",            &Action::get_proxies },
    { "connect_accelerator",    &Action::connect_accelerator },
    { "disconnect_accelerator", &Action::disconnect_accelerator },
#if GTK_CHECK_VERSION( 2, 16, 0 )
    { "block_activate",         &Action::block_activate },
    { "unblock_activate",       &Action::unblock_activate },
#endif
#if 0 // deprecated
    { "block_activate_from",    &Action::block_activate_from },
    { "unblock_activate_from",  &Action::unblock_activate_from },
#endif
#if GTK_CHECK_VERSION( 2, 20, 0 )
    { "get_always_show_image",  &Action::get_always_show_image },
    { "set_always_show_image",  &Action::set_always_show_image },
#endif
    { "get_accel_path",         &Action::get_accel_path },
    { "set_accel_path",         &Action::set_accel_path },
    //{ "get_accel_closure",     &Action::get_accel_closure },
    //{ "set_accel_group",       &Action::set_accel_group },
#if GTK_CHECK_VERSION( 2, 16, 0 )
    { "set_label",              &Action::set_label },
    { "get_label",              &Action::get_label },
    { "set_short_label",        &Action::set_short_label },
    { "get_short_label",        &Action::get_short_label },
    { "set_tooltip",            &Action::set_tooltip },
    { "get_tooltip",            &Action::get_tooltip },
    { "set_stock_id",           &Action::set_stock_id },
    { "get_stock_id",           &Action::get_stock_id },
    //{ "set_gicon",              &Action::set_gicon },
    //{ "get_gicon",              &Action::get_gicon },
    { "set_icon_name",          &Action::set_icon_name },
    { "get_icon_name",          &Action::get_icon_name },
    { "set_visible_horizontal", &Action::set_visible_horizontal },
    { "get_visible_horizontal", &Action::get_visible_horizontal },
    { "set_visible_vertical",   &Action::set_visible_vertical },
    { "get_visible_vertical",   &Action::get_visible_vertical },
    { "set_is_important",       &Action::set_is_important },
    { "get_is_important",       &Action::get_is_important },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Action, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_Action );
}


Action::Action( const Falcon::CoreClass* gen, const GtkAction* act )
    :
    Gtk::CoreGObject( gen, (GObject*) act )
{}


Falcon::CoreObject* Action::factory( const Falcon::CoreClass* gen, void* act, bool )
{
    return new Action( gen, (GtkAction*) act );
}


/*#
    @class GtkAction
    @brief An action which can be triggered by a menu or toolbar item
    @param name A unique name for the action
    @optparam label the label displayed in menu items and on buttons
    @optparam tooltip a tooltip for the action
    @optparam stock_id the stock icon to display in widgets representing the action

    Actions represent operations that the user can be perform, along with some
    information how it should be presented in the interface. Each action provides
    methods to create icons, menu items and toolbar items representing itself.

    As well as the callback that is called when the action gets activated, the
    following also gets associated with the action:

    - a name (not translated, for path lookup)
    - a label (translated, for display)
    - an accelerator
    - whether label indicates a stock id
    - a tooltip (optional, translated)
    - a toolbar label (optional, shorter than label)

    The action will also have some state information:

    - visible (shown/hidden)
    - sensitive (enabled/disabled)

    Apart from regular actions, there are toggle actions, which can be toggled
    between two states and radio actions, of which only one in a group can be in
    the "active" state. Other actions can be implemented as GtkAction subclasses.

    Each action can have one or more proxy menu item, toolbar button or other
    proxy widgets. Proxies mirror the state of the action (text label, tooltip,
    icon, visible, sensitive, etc), and should change when the action's state
    changes. When the proxy is activated, it should activate its action.
 */
FALCON_FUNC Action::init( VMARG )
{
    MYSELF;

    if ( self->getObject() )
        return;

    Gtk::ArgCheck4 args( vm, "S[,S,S,S]" );
    const char* nam = args.getCString( 0 );
    const char* lbl = args.getCString( 1, false );
    const char* tooltip = args.getCString( 2, false );
    const char* stock_id = args.getCString( 3, false );

    GtkAction* act = gtk_action_new( nam, lbl, tooltip, stock_id );
    self->setObject( (GObject*) act );
}


/*#
    @method signal_activate GtkAction
    @brief The "activate" signal is emitted when the action is activated.
 */
FALCON_FUNC Action::signal_activate( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "activate", (void*) &Action::on_activate, vm );
}


void Action::on_activate( GtkAction* act, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) act, "activate", "on_activate", (VMachine*)_vm );
}


/*#
    @method get_name GtkAction
    @brief Returns the name of the action.
    @return the name of the action
 */
FALCON_FUNC Action::get_name( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* nam = gtk_action_get_name( (GtkAction*)_obj );
    if ( nam )
        vm->retval( UTF8String( nam ) );
    else
        vm->retnil();
}


/*#
    @method is_sensitive GtkAction
    @brief Returns whether the action is effectively sensitive.
    @return true if the action and its associated action group are both sensitive.
 */
FALCON_FUNC Action::is_sensitive( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_action_is_sensitive( (GtkAction*)_obj ) );
}


/*#
    @method get_sensitive GtkAction
    @brief Returns whether the action itself is sensitive.
    @return true if the action itself is sensitive.

    Note that this doesn't necessarily mean effective sensitivity.
    See is_sensitive() for that.
 */
FALCON_FUNC Action::get_sensitive( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_action_get_sensitive( (GtkAction*)_obj ) );
}


/*#
    @method set_sensitive GtkAction
    @brief Sets the sensitive property of the action to sensitive.
    @param sensitive true to make the action sensitive

    Note that this doesn't necessarily mean effective sensitivity.
    See is_sensitive() for that.
 */
FALCON_FUNC Action::set_sensitive( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_set_sensitive( (GtkAction*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method is_visible GtkAction
    @brief Returns whether the action is effectively visible.
    @return true if the action and its associated action group are both visible.
 */
FALCON_FUNC Action::is_visible( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_action_is_visible( (GtkAction*)_obj ) );
}


/*#
    @method get_visible GtkAction
    @brief Returns whether the action itself is visible.
    @return true if the action itself is visible.

    Note that this doesn't necessarily mean effective visibility.
    See is_sensitive() for that.
 */
FALCON_FUNC Action::get_visible( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_action_get_visible( (GtkAction*)_obj ) );
}


/*#
    @method set_visible GtkAction
    @brief Sets the visible property of the action to visible.
    @param true to make the action visible

    Note that this doesn't necessarily mean effective visibility.
    See is_visible() for that.
 */
FALCON_FUNC Action::set_visible( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_set_visible( (GtkAction*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method activate GtkAction
    @brief Emits the "activate" signal on the specified action, if it isn't insensitive.

    This gets called by the proxy widgets when they get activated.

    It can also be used to manually activate an action.
 */
FALCON_FUNC Action::activate( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_action_activate( (GtkAction*)_obj );
}


/*#
    @method create_icon GtkAction
    @brief This function is intended for use by action implementations to create icons displayed in the proxy widgets.
    @param icon_size the size of the icon that should be created (GtkIconSize).
    @return a widget that displays the icon for this action (GtkImage).
 */
FALCON_FUNC Action::create_icon( VMARG )
{
    Item* i_sz = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sz || !i_sz->isInteger() )
        throw_inv_params( "GtkIconSize" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* ico = gtk_action_create_icon( (GtkAction*)_obj,
                                             (GtkIconSize) i_sz->asInteger() );
    vm->retval( new Gtk::Image( vm->findWKI( "GtkImage" )->asClass(), (GtkImage*) ico ) );
}


/*#
    @method create_menu_item GtkAction
    @brief Creates a menu item widget that proxies for the given action.
    @return a menu item connected to the action (GtkMenuItem).
 */
FALCON_FUNC Action::create_menu_item( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* itm = gtk_action_create_menu_item( (GtkAction*)_obj );
    vm->retval( new Gtk::MenuItem( vm->findWKI( "GtkMenuItem" )->asClass(),
                                   (GtkMenuItem*) itm ) );
}


/*#
    @method create_tool_item GtkAction
    @brief Creates a toolbar item widget that proxies for the given action.
    @return a toolbar item connected to the action (GtkToolItem).
 */
FALCON_FUNC Action::create_tool_item( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* itm = gtk_action_create_tool_item( (GtkAction*)_obj );
    vm->retval( new Gtk::ToolItem( vm->findWKI( "GtkToolItem" )->asClass(),
                                   (GtkToolItem*) itm ) );
}


/*#
    @method create_menu GtkAction
    @brief If action provides a GtkMenu widget as a submenu for the menu item or the toolbar item it creates, this function returns an instance of that menu.
    @return the menu item provided by the action (GtkMenu), or nil.
 */
FALCON_FUNC Action::create_menu( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* menu = gtk_action_create_menu( (GtkAction*)_obj );
    if ( menu )
        vm->retval( new Gtk::Menu( vm->findWKI( "GtkMenu" )->asClass(), (GtkMenu*) menu ) );
    else
        vm->retnil();
}


#if 0 // deprecated
FALCON_FUNC Action::connect_proxy( VMARG );
FALCON_FUNC Action::disconnect_proxy( VMARG );
#endif


/*#
    @method get_proxies GtkAction
    @brief Returns the proxy widgets for an action.
    @return an array of proxy widgets (GtkWidget).

    See also gtk_widget_get_action().
 */
FALCON_FUNC Action::get_proxies( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GSList* lst = gtk_action_get_proxies( (GtkAction*)_obj );
    int cnt = 0;
    GSList* el;
    for ( el = lst; el; el = el->next ) ++cnt;
    CoreArray* arr = new CoreArray( cnt );
    if ( cnt )
    {
        Item* wki = vm->findWKI( "GtkWidget" );
        for ( el = lst; el; el = el->next )
            arr->append( new Gtk::Widget( wki->asClass(), (GtkWidget*) el->data ) );
    }
    vm->retval( arr );
}


/*#
    @method connect_accelerator GtkAction
    @brief Installs the accelerator for action if action has an accel path and group.

    See set_accel_path() and set_accel_group()

    Since multiple proxies may independently trigger the installation of the accelerator,
    the action counts the number of times this function has been called and doesn't
    remove the accelerator until disconnect_accelerator() has been called as many times.
 */
FALCON_FUNC Action::connect_accelerator( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_action_connect_accelerator( (GtkAction*)_obj );
}


/*#
    @method disconnect_accelerator GtkAction
    @brief Undoes the effect of one call to connect_accelerator().
 */
FALCON_FUNC Action::disconnect_accelerator( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_action_disconnect_accelerator( (GtkAction*)_obj );
}


#if GTK_CHECK_VERSION( 2, 16, 0 )
/*#
    @method block_activate GtkAction
    @brief Disable activation signals from the action

    This is needed when updating the state of your proxy GtkActivatable widget could
    result in calling gtk_action_activate(), this is a convenience function to avoid
    recursing in those cases (updating toggle state for instance).
 */
FALCON_FUNC Action::block_activate( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_action_block_activate( (GtkAction*)_obj );
}


/*#
    @method unblock_activate GtkAction
    @brief Reenable activation signals from the action
 */
FALCON_FUNC Action::unblock_activate( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_action_unblock_activate( (GtkAction*)_obj );
}
#endif // GTK_CHECK_VERSION( 2, 16, 0 )


#if 0 // deprecated
FALCON_FUNC Action::block_activate_from( VMARG );
FALCON_FUNC Action::unblock_activate_from( VMARG );
#endif


#if GTK_CHECK_VERSION( 2, 20, 0 )
/*#
    @method get_always_show_image GtkAction
    @brief Returns whether action's menu item proxies will ignore the "gtk-menu-images" setting and always show their image, if available.
    @return true if the menu item proxies will always show their image
 */
FALCON_FUNC Action::get_always_show_image( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_action_get_always_show_image( (GtkAction*)_obj ) );
}


/*#
    @method set_always_show_image
    @brief Sets whether action's menu item proxies will ignore the "gtk-menu-images" setting and always show their image, if available.

    Use this if the menu item would be useless or hard to use without their image.
 */
FALCON_FUNC Action::set_always_show_image( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_set_always_show_image( (GtkAction*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}
#endif // GTK_CHECK_VERSION( 2, 20, 0 )


/*#
    @method get_accel_path GtkAction
    @brief Returns the accel path for this action.
    @return the accel path for this action, or nil if none is set.
 */
FALCON_FUNC Action::get_accel_path( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* path = gtk_action_get_accel_path( (GtkAction*)_obj );
    if ( path )
        vm->retval( UTF8String( path ) );
    else
        vm->retnil();
}


/*#
    @method set_accel_path GtkAction
    @brief Sets the accel path for this action.
    @param accel_path the accelerator path (string)

    All proxy widgets associated with the action will have this accel path, so that
    their accelerators are consistent.
 */
FALCON_FUNC Action::set_accel_path( VMARG )
{
    Item* i_path = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || i_path->isNil() || !i_path->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString path( i_path->asString() );
    gtk_action_set_accel_path( (GtkAction*)_obj, path.c_str() );
}


//FALCON_FUNC Action::get_accel_closure( VMARG );
//FALCON_FUNC Action::set_accel_group( VMARG );


#if GTK_CHECK_VERSION( 2, 16, 0 )
/*#
    @method set_label GtkAction
    @brief Sets the label of action.
    @param label the label text to set
 */
FALCON_FUNC Action::set_label( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || i_lbl->isNil() || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString lbl( i_lbl->asString() );
    gtk_action_set_label( (GtkAction*)_obj, lbl.c_str() );
}


/*#
    @method get_label GtkAction
    @brief Gets the label text of action.
    @return the label text
 */
FALCON_FUNC Action::get_label( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_action_get_label( (GtkAction*)_obj );
    if ( lbl )
        vm->retval( UTF8String( lbl ) );
    else
        vm->retnil();
}


/*#
    @method set_short_label GtkAction
    @brief Sets a shorter label text on action.
    @param short_label the label text to set
 */
FALCON_FUNC Action::set_short_label( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || i_lbl->isNil() || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString lbl( i_lbl->asString() );
    gtk_action_set_short_label( (GtkAction*)_obj, lbl.c_str() );
}


/*#
    @method get_short_label GtkAction
    @brief Gets the short label text of action.
    @return the short label text
 */
FALCON_FUNC Action::get_short_label( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_action_get_short_label( (GtkAction*)_obj );
    if ( lbl )
        vm->retval( UTF8String( lbl ) );
    else
        vm->retnil();
}


/*#
    @method set_tooltip GtkAction
    @brief Sets the tooltip text on action
    @param tooltip the tooltip text
 */
FALCON_FUNC Action::set_tooltip( VMARG )
{
    Item* i_tip = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tip || i_tip->isNil() || !i_tip->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString tip( i_tip->asString() );
    gtk_action_set_tooltip( (GtkAction*)_obj, tip.c_str() );
}


/*#
    @method get_tooltip GtkAction
    @brief Gets the tooltip text of action.
    @return the tooltip text
 */
FALCON_FUNC Action::get_tooltip( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* tip = gtk_action_get_tooltip( (GtkAction*)_obj );
    if ( tip )
        vm->retval( UTF8String( tip ) );
    else
        vm->retnil();
}


/*#
    @method set_stock_id GtkAction
    @brief Sets the stock id on action
    @param stock_id the stock id
 */
FALCON_FUNC Action::set_stock_id( VMARG )
{
    Item* i_id = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || i_id->isNil() || !i_id->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString stid( i_id->asString() );
    gtk_action_set_stock_id( (GtkAction*)_obj, stid.c_str() );
}


/*#
    @method get_stock_id GtkAction
    @brief Gets the stock id of action.
    @return the stock id
 */
FALCON_FUNC Action::get_stock_id( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* id = gtk_action_get_stock_id( (GtkAction*)_obj );
    if ( id )
        vm->retval( UTF8String( id ) );
    else
        vm->retnil();
}


//FALCON_FUNC Action::set_gicon( VMARG );
//FALCON_FUNC Action::get_gicon( VMARG );


/*#
    @method set_icon_name GtkAction
    @brief Sets the icon name on action
    @param icon_name the icon name to set
 */
FALCON_FUNC Action::set_icon_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* nm = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_action_set_icon_name( (GtkAction*)_obj, nm );
}


/*#
    @method get_icon_name GtkAction
    @brief Gets the icon name of action.
    @return the icon name.
 */
FALCON_FUNC Action::get_icon_name( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* nm = gtk_action_get_icon_name( (GtkAction*)_obj );
    if ( nm )
        vm->retval( UTF8String( nm ) );
    else
        vm->retnil();
}


/*#
    @method set_visible_horizontal GtkAction
    @brief Sets whether action is visible when horizontal
    @param visible_horizontal whether the action is visible horizontally
 */
FALCON_FUNC Action::set_visible_horizontal( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_set_visible_horizontal( (GtkAction*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_visible_horizontal GtkAction
    @brief Checks whether action is visible when horizontal
    @return whether action is visible when horizontal
 */
FALCON_FUNC Action::get_visible_horizontal( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_action_get_visible_horizontal( (GtkAction*)_obj ) );
}


/*#
    @method set_visible_vertical GtkAction
    @brief Sets whether action is visible when vertical
    @param visible_vertical whether the action is visible vertically
 */
FALCON_FUNC Action::set_visible_vertical( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_set_visible_vertical( (GtkAction*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_visible_vertical GtkAction
    @brief Checks whether action is visible when horizontal
    @return whether action is visible when horizontal
 */
FALCON_FUNC Action::get_visible_vertical( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_action_get_visible_vertical( (GtkAction*)_obj ) );
}


/*#
    @method set_is_important GtkAction
    @brief Sets whether the action is important.
    @param is_important TRUE to make the action important

    This attribute is used primarily by toolbar items to decide whether to show
    a label or not.
 */
FALCON_FUNC Action::set_is_important( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_set_is_important( (GtkAction*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_is_important GtkAction
    @brief Checks whether action is important or not
    @return whether action is important
 */
FALCON_FUNC Action::get_is_important( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_action_get_is_important( (GtkAction*)_obj ) );
}

#endif // GTK_CHECK_VERSION( 2, 16, 0 )

} // Gtk
} // Falcon
