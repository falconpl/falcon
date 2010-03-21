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

    Gtk::MethodTab methods[] =
    {
    { "signal_activate",        &Action::signal_activate },
    //{ "get_name",     &Action::foo },
    //{ "is_sensitive",     &Action::foo },
    //{ "get_sensitive",     &Action::foo },
    //{ "set_sensitive",     &Action::foo },
    //{ "is_visible",     &Action::foo },
    //{ "get_visible",     &Action::foo },
    //{ "set_visible",     &Action::foo },
    //{ "activate",     &Action::foo },
    //{ "create_icon",     &Action::foo },
    //{ "create_menu_item",     &Action::foo },
    //{ "create_tool_item",     &Action::foo },
    //{ "create_menu",     &Action::foo },
    //{ "connect_proxy",     &Action::foo },
    //{ "disconnect_proxy",     &Action::foo },
    //{ "get_proxies",     &Action::foo },
    //{ "connect_accelerator",     &Action::foo },
    //{ "disconnect_accelerator",     &Action::foo },
    //{ "block_activate",     &Action::foo },
    //{ "unblock_activate",     &Action::foo },
    //{ "block_activate_from",     &Action::foo },
    //{ "unblock_activate_from",     &Action::foo },
    //{ "get_always_show_image",     &Action::foo },
    //{ "set_always_show_image",     &Action::foo },
    //{ "get_accel_path",     &Action::foo },
    //{ "set_accel_path",     &Action::foo },
    //{ "get_accel_closure",     &Action::foo },
    //{ "set_accel_group",     &Action::foo },
    //{ "set_label",     &Action::foo },
    //{ "get_label",     &Action::foo },
    //{ "set_short_label",     &Action::foo },
    //{ "get_short_label",     &Action::foo },
    //{ "set_tooltip",     &Action::foo },
    //{ "get_tooltip",     &Action::foo },
    //{ "set_stock_id",     &Action::foo },
    //{ "get_stock_id",     &Action::foo },
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
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Action, meth->name, meth->cb );
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


//FALCON_FUNC Action::get_name( VMARG );
//FALCON_FUNC Action::is_sensitive( VMARG );
//FALCON_FUNC Action::get_sensitive( VMARG );
//FALCON_FUNC Action::set_sensitive( VMARG );
//FALCON_FUNC Action::is_visible( VMARG );
//FALCON_FUNC Action::get_visible( VMARG );
//FALCON_FUNC Action::set_visible( VMARG );
//FALCON_FUNC Action::activate( VMARG );
//FALCON_FUNC Action::create_icon( VMARG );
//FALCON_FUNC Action::create_menu_item( VMARG );
//FALCON_FUNC Action::create_tool_item( VMARG );
//FALCON_FUNC Action::create_menu( VMARG );
//FALCON_FUNC Action::connect_proxy( VMARG );
//FALCON_FUNC Action::disconnect_proxy( VMARG );
//FALCON_FUNC Action::get_proxies( VMARG );
//FALCON_FUNC Action::connect_accelerator( VMARG );
//FALCON_FUNC Action::disconnect_accelerator( VMARG );
//FALCON_FUNC Action::block_activate( VMARG );
//FALCON_FUNC Action::unblock_activate( VMARG );
//FALCON_FUNC Action::block_activate_from( VMARG );
//FALCON_FUNC Action::unblock_activate_from( VMARG );
//FALCON_FUNC Action::get_always_show_image( VMARG );
//FALCON_FUNC Action::set_always_show_image( VMARG );
//FALCON_FUNC Action::get_accel_path( VMARG );
//FALCON_FUNC Action::set_accel_path( VMARG );
//FALCON_FUNC Action::get_accel_closure( VMARG );
//FALCON_FUNC Action::set_accel_group( VMARG );
//FALCON_FUNC Action::set_label( VMARG );
//FALCON_FUNC Action::get_label( VMARG );
//FALCON_FUNC Action::set_short_label( VMARG );
//FALCON_FUNC Action::get_short_label( VMARG );
//FALCON_FUNC Action::set_tooltip( VMARG );
//FALCON_FUNC Action::get_tooltip( VMARG );
//FALCON_FUNC Action::set_stock_id( VMARG );
//FALCON_FUNC Action::get_stock_id( VMARG );
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


} // Gtk
} // Falcon
