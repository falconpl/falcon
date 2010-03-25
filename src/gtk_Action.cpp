/**
 *  \file gtk_Action.cpp
 */

#include "gtk_Action.hpp"

#include <gtk/gtk.h>


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
    //{ "create_icon",     &Action::foo },
    //{ "create_menu_item",     &Action::foo },
    //{ "create_tool_item",     &Action::foo },
    //{ "create_menu",     &Action::foo },
    //{ "connect_proxy",     &Action::foo },
    //{ "disconnect_proxy",     &Action::foo },
    //{ "get_proxies",     &Action::foo },
    { "connect_accelerator",    &Action::connect_accelerator },
    { "disconnect_accelerator", &Action::disconnect_accelerator },
#if GTK_MINOR_VERSION >= 16
    { "block_activate",         &Action::block_activate },
    { "unblock_activate",       &Action::unblock_activate },
#endif
    //{ "block_activate_from",     &Action::foo },
    //{ "unblock_activate_from",     &Action::foo },
#if GTK_MINOR_VERSION >= 20
    { "get_always_show_image",  &Action::get_always_show_image },
    { "set_always_show_image",  &Action::set_always_show_image },
#endif
    { "get_accel_path",         &Action::get_accel_path },
    { "set_accel_path",         &Action::set_accel_path },
    //{ "get_accel_closure",     &Action::foo },
    //{ "set_accel_group",     &Action::foo },
#if GTK_MINOR_VERSION >= 16
    { "set_label",              &Action::set_label },
    { "get_label",              &Action::get_label },
    { "set_short_label",        &Action::set_short_label },
    { "get_short_label",        &Action::get_short_label },
    { "set_tooltip",            &Action::set_tooltip },
    { "get_tooltip",            &Action::get_tooltip },
    { "set_stock_id",           &Action::set_stock_id },
    { "get_stock_id",           &Action::get_stock_id },
    //{ "set_gicon",     &Action::foo },
    //{ "get_gicon",     &Action::foo },
    //{ "set_icon_name",     &Action::foo },
    //{ "get_icon_name",     &Action::foo },
    //{ "set_visible_horizontal",     &Action::foo },
    //{ "get_visible_horizontal",     &Action::foo },
    //{ "set_visible_vertical",     &Action::foo },
    //{ "get_visible_vertical",     &Action::foo },
    //{ "set_is_important",     &Action::foo },
    //{ "get_is_important",     &Action::foo },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Action, meth->name, meth->cb );
}


Action::Action( const Falcon::CoreClass* gen, const GtkAction* act )
    :
    Gtk::CoreGObject( gen )
{
    if ( act )
        setUserData( new GData( (GObject*) act ) );
}


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

    if ( self->getUserData() )
        return;

    Gtk::ArgCheck<4> args( vm, "S[,S,S,S]" );
    const char* nam = args.getCString( 0 );
    const char* lbl = args.getCString( 1, false );
    const char* tooltip = args.getCString( 2, false );
    const char* stock_id = args.getCString( 3, false );

    GtkAction* act = gtk_action_new( nam, lbl, tooltip, stock_id );
    Gtk::internal_add_slot( (GObject*) act );
    self->setUserData( new GData( (GObject*) act ) );
}


/*#
    @method signal_activate GtkAction
    @brief Connect a VMSlot to the action activate signal and return it

    The "activate" signal is emitted when the action is activated.
 */
FALCON_FUNC Action::signal_activate( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "activate", (void*) &Action::on_activate, vm );
}

void Action::on_activate( GtkAction* act, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) act, "activate", "on_activate", (VMachine*)_vm );
}


/*#
    @method get_name GtkAction
    @brief Returns the name of the action.
    @return the name of the action
 */
FALCON_FUNC Action::get_name( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* nam = gtk_action_get_name( (GtkAction*)_obj );
    vm->retval( nam ? new String( nam ) : new String );
}


/*#
    @method is_sensitive GtkAction
    @brief Returns whether the action is effectively sensitive.
    @return true if the action and its associated action group are both sensitive.
 */
FALCON_FUNC Action::is_sensitive( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_activate( (GtkAction*)_obj );
}


//FALCON_FUNC Action::create_icon( VMARG );
//FALCON_FUNC Action::create_menu_item( VMARG );
//FALCON_FUNC Action::create_tool_item( VMARG );
//FALCON_FUNC Action::create_menu( VMARG );
//FALCON_FUNC Action::connect_proxy( VMARG );
//FALCON_FUNC Action::disconnect_proxy( VMARG );
//FALCON_FUNC Action::get_proxies( VMARG );


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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_disconnect_accelerator( (GtkAction*)_obj );
}


#if GTK_MINOR_VERSION >= 16
/*#
    @method block_activate GtkAction
    @brief Disable activation signals from the action

    This is needed when updating the state of your proxy GtkActivatable widget could
    result in calling gtk_action_activate(), this is a convenience function to avoid
    recursing in those cases (updating toggle state for instance).
 */
FALCON_FUNC Action::block_activate( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_action_unblock_activate( (GtkAction*)_obj );
}
#endif // GTK_MINOR_VERSION >= 16

//FALCON_FUNC Action::block_activate_from( VMARG );
//FALCON_FUNC Action::unblock_activate_from( VMARG );


#if GTK_MINOR_VERSION >= 20
/*#
    @method get_always_show_image GtkAction
    @brief Returns whether action's menu item proxies will ignore the "gtk-menu-images" setting and always show their image, if available.
    @return true if the menu item proxies will always show their image
 */
FALCON_FUNC Action::get_always_show_image( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
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
#endif // GTK_MINOR_VERSION >= 20


/*#
    @method get_accel_path GtkAction
    @brief Returns the accel path for this action.
    @return the accel path for this action, or nil if none is set.
 */
FALCON_FUNC Action::get_accel_path( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* path = gtk_action_get_accel_path( (GtkAction*)_obj );
    if ( path )
        vm->retval( new String( path ) );
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


#if GTK_MINOR_VERSION >= 16
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_action_get_label( (GtkAction*)_obj );
    if ( lbl )
        vm->retval( new String( lbl ) );
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_action_get_short_label( (GtkAction*)_obj );
    if ( lbl )
        vm->retval( new String( lbl ) );
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* tip = gtk_action_get_tooltip( (GtkAction*)_obj );
    if ( tip )
        vm->retval( new String( tip ) );
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
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* id = gtk_action_get_stock_id( (GtkAction*)_obj );
    if ( id )
        vm->retval( new String( id ) );
    else
        vm->retnil();
}


//FALCON_FUNC Action::set_gicon( VMARG );
//FALCON_FUNC Action::get_gicon( VMARG );
//FALCON_FUNC Action::set_icon_name( VMARG );
//FALCON_FUNC Action::get_icon_name( VMARG );
//FALCON_FUNC Action::set_visible_horizontal( VMARG );
//FALCON_FUNC Action::get_visible_horizontal( VMARG );
//FALCON_FUNC Action::set_visible_vertical( VMARG );
//FALCON_FUNC Action::get_visible_vertical( VMARG );
//FALCON_FUNC Action::set_is_important( VMARG );
//FALCON_FUNC Action::get_is_important( VMARG );

#endif // GTK_MINOR_VERSION >= 16

} // Gtk
} // Falcon
