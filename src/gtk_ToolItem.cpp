/**
 *  \file gtk_ToolItem.cpp
 */

#include "gtk_ToolItem.hpp"

#include "gtk_Buildable.hpp"
#include "gtk_Activatable.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ToolItem::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ToolItem = mod->addClass( "GtkToolItem", &ToolItem::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_ToolItem->getClassDef()->addInheritance( in );

    c_ToolItem->setWKS( true );
    c_ToolItem->getClassDef()->factory( &ToolItem::factory );

    Gtk::MethodTab methods[] =
    {
#if 0 // todo
    { "signal_create_menu_proxy",   &ToolItem::signal_create_menu_proxy },
    { "signal_set_tooltip",         &ToolItem::signal_set_tooltip },
    { "signal_toolbar_reconfigured",&ToolItem::signal_toolbar_reconfigured },
#endif
    { "set_homogeneous",            &ToolItem::set_homogeneous },
    { "get_homogeneous",            &ToolItem::get_homogeneous },
    { "set_expand",                 &ToolItem::set_expand },
    { "get_expand",                 &ToolItem::get_expand },
    //{ "set_tooltip",              &ToolItem::set_tooltip },
    { "set_tooltip_text",           &ToolItem::set_tooltip_text },
    { "set_tooltip_markup",         &ToolItem::set_tooltip_markup },
    { "set_use_drag_window",        &ToolItem::set_use_drag_window },
    { "get_use_drag_window",        &ToolItem::get_use_drag_window },
    { "set_visible_horizontal",     &ToolItem::set_visible_horizontal },
    { "get_visible_horizontal",     &ToolItem::get_visible_horizontal },
    { "set_visible_vertical",       &ToolItem::set_visible_vertical },
    { "get_visible_vertical",       &ToolItem::get_visible_vertical },
    { "set_is_important",           &ToolItem::set_is_important },
    { "get_is_important",           &ToolItem::get_is_important },
#if GTK_CHECK_VERSION( 2, 20, 0 )
    { "get_ellipsize_mode",         &ToolItem::get_ellipsize_mode },
#endif
    { "get_icon_size",              &ToolItem::get_icon_size },
    { "get_orientation",            &ToolItem::get_orientation },
    { "get_toolbar_style",          &ToolItem::get_toolbar_style },
    { "get_relief_style",           &ToolItem::get_relief_style },
#if GTK_CHECK_VERSION( 2, 20, 0 )
    { "get_text_alignment",         &ToolItem::get_text_alignment },
    { "get_text_orientation",       &ToolItem::get_text_orientation },
#endif
    //{ "retrieve_proxy_menu_item", &ToolItem::retrieve_proxy_menu_item },
    //{ "get_proxy_menu_item",      &ToolItem::get_proxy_menu_item },
    { "set_proxy_menu_item",        &ToolItem::set_proxy_menu_item },
    { "rebuild_menu",               &ToolItem::rebuild_menu },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "toolbar_reconfigured",       &ToolItem::toolbar_reconfigured },
#endif
    //{ "get_text_size_group",      &ToolItem::get_text_size_group },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ToolItem, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_ToolItem );
#if GTK_CHECK_VERSION( 2, 16, 0 )
    Gtk::Activatable::clsInit( mod, c_ToolItem );
#endif

}


ToolItem::ToolItem( const Falcon::CoreClass* gen, const GtkToolItem* itm )
    :
    Gtk::CoreGObject( gen, (GObject*) itm )
{}


Falcon::CoreObject* ToolItem::factory( const Falcon::CoreClass* gen, void* itm, bool )
{
    return new ToolItem( gen, (GtkToolItem*) itm );
}


/*#
    @class GtkToolItem
    @brief The base class of widgets that can be added to GtkToolShell

    GtkToolItems are widgets that can appear on a toolbar. To create a toolbar
    item that contain something else than a button, use gtk_tool_item_new().
    Use gtk_container_add() to add a child widget to the tool item.

    For toolbar items that contain buttons, see the GtkToolButton,
    GtkToggleToolButton and GtkRadioToolButton classes.

    See the GtkToolbar class for a description of the toolbar widget,
    and GtkToolShell for a description of the tool shell interface.
 */
FALCON_FUNC ToolItem::init( VMARG )
{
    MYSELF;
    if ( self->getObject() )
        return;
    NO_ARGS
    self->setObject( (GObject*) gtk_tool_item_new() );
}


#if 0
FALCON_FUNC ToolItem::signal_create_menu_proxy( VMARG );
FALCON_FUNC ToolItem::signal_set_tooltip( VMARG );
FALCON_FUNC ToolItem::signal_toolbar_reconfigured( VMARG );
#endif


/*#
    @method set_homogeneous GtkToolItem
    @brief Sets whether tool_item is to be allocated the same size as other homogeneous items.

    The effect is that all homogeneous items will have the same width as the widest of the items.
 */
FALCON_FUNC ToolItem::set_homogeneous( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_homogeneous( (GtkToolItem*)_obj,
                                   i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_homogeneous GtkToolItem
    @brief Returns whether tool_item is the same size as other homogeneous items.
    @return TRUE if the item is the same size as other homogeneous items.
 */
FALCON_FUNC ToolItem::get_homogeneous( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_item_get_homogeneous( (GtkToolItem*)_obj ) );
}


/*#
    @method set_expand GtkToolItem
    @brief Sets whether tool_item is allocated extra space when there is more room on the toolbar then needed for the items.
    @param expand Whether tool_item is allocated extra space

    The effect is that the item gets bigger when the toolbar gets bigger and
    smaller when the toolbar gets smaller.
 */
FALCON_FUNC ToolItem::set_expand( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_expand( (GtkToolItem*)_obj,
                              i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_expand GtkToolItem
    @brief Returns whether tool_item is allocated extra space.
    @return TRUE if tool_item is allocated extra space.
 */
FALCON_FUNC ToolItem::get_expand( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_item_get_expand( (GtkToolItem*)_obj ) );
}


//FALCON_FUNC ToolItem::set_tooltip( VMARG );


/*#
    @method set_tooltip_text GtkToolItem
    @brief Sets the text to be displayed as tooltip on the item.
    @param text text to be used as tooltip for tool_item
 */
FALCON_FUNC ToolItem::set_tooltip_text( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* tip = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_tooltip_text( (GtkToolItem*)_obj, tip );
}


/*#
    @method set_tooltip_markup GtkToolItem
    @brief Sets the markup text to be displayed as tooltip on the item.
    @param markup markup text to be used as tooltip for tool_item
 */
FALCON_FUNC ToolItem::set_tooltip_markup( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* mk = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_tooltip_markup( (GtkToolItem*)_obj, mk );
}


/*#
    @method set_use_drag_window GtkToolItem
    @brief Sets whether tool_item has a drag window.
    @param use_drag_window Whether tool_item has a drag window.

    When TRUE the toolitem can be used as a drag source through gtk_drag_source_set().
    When tool_item has a drag window it will intercept all events, even those
    that would otherwise be sent to a child of tool_item.
 */
FALCON_FUNC ToolItem::set_use_drag_window( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_use_drag_window( (GtkToolItem*)_obj,
                                       i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_use_drag_window GtkToolItem
    @brief Returns whether tool_item has a drag window.
    @return TRUE if tool_item uses a drag window.
 */
FALCON_FUNC ToolItem::get_use_drag_window( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_item_get_use_drag_window( (GtkToolItem*)_obj ) );
}


/*#
    @method set_visible_horizontal GtkToolItem
    @brief Sets whether tool_item is visible when the toolbar is docked horizontally.
    @param visible_horizontal Whether tool_item is visible when in horizontal mode
 */
FALCON_FUNC ToolItem::set_visible_horizontal( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_visible_horizontal( (GtkToolItem*)_obj,
                                          i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_visible_horizontal GtkToolItem
    @brief Returns whether the tool_item is visible on toolbars that are docked horizontally.
    @return TRUE if tool_item is visible on toolbars that are docked horizontally.
 */
FALCON_FUNC ToolItem::get_visible_horizontal( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_item_get_visible_horizontal( (GtkToolItem*)_obj ) );
}


/*#
    @method set_visible_vertical GtkToolItem
    @brief Sets whether tool_item is visible when the toolbar is docked vertically.
    @param visible_vertical whether tool_item is visible when the toolbar is in vertical mode

    Some tool items, such as text entries, are too wide to be useful on a
    vertically docked toolbar. If visible_vertical is FALSE tool_item will not
    appear on toolbars that are docked vertically.
 */
FALCON_FUNC ToolItem::set_visible_vertical( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_visible_vertical( (GtkToolItem*)_obj,
                                        i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_visible_vertical GtkToolItem
    @brief Returns whether tool_item is visible when the toolbar is docked vertically.
    @return Whether tool_item is visible when the toolbar is docked vertically
 */
FALCON_FUNC ToolItem::get_visible_vertical( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_item_get_visible_vertical( (GtkToolItem*)_obj ) );
}


/*#
    @method set_is_important GtkToolItem
    @brief Sets whether tool_item should be considered important.
    @param is_important whether the tool item should be considered important

    The GtkToolButton class uses this property to determine whether to show or
    hide its label when the toolbar style is GTK_TOOLBAR_BOTH_HORIZ. The result
    is that only tool buttons with the "is_important" property set have labels,
    an effect known as "priority text".
 */
FALCON_FUNC ToolItem::set_is_important( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_is_important( (GtkToolItem*)_obj,
                                    i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_is_important GtkToolItem
    @brief Returns whether tool_item is considered important.
    @return TRUE if tool_item is considered important.
 */
FALCON_FUNC ToolItem::get_is_important( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_item_get_is_important( (GtkToolItem*)_obj ) );
}


#if GTK_CHECK_VERSION( 2, 20, 0 )
/*#
    @method get_ellipsize_mode GtkToolItem
    @brief Returns the ellipsize mode used for tool_item.
    @return a PangoEllipsizeMode indicating how text in tool_item should be ellipsized.

    Custom subclasses of GtkToolItem should call this function to find out how
    text should be ellipsized.
 */
FALCON_FUNC ToolItem::get_ellipsize_mode( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_item_get_ellipsize_mode( (GtkToolItem*)_obj ) );
}
#endif


/*#
    @method get_icon_size GtkToolItem
    @brief Returns the icon size used for tool_item.
    @return a GtkIconSize indicating the icon size used for tool_item.

    Custom subclasses of GtkToolItem should call this function to find out what
    size icons they should use.
 */
FALCON_FUNC ToolItem::get_icon_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_item_get_icon_size( (GtkToolItem*)_obj ) );
}


/*#
    @method get_orientation GtkToolItem
    @brief Returns the orientation used for tool_item.
    @return a GtkOrientation indicating the orientation used for tool_item

    Custom subclasses of GtkToolItem should call this function to find out what
    size icons they should use.
 */
FALCON_FUNC ToolItem::get_orientation( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_item_get_orientation( (GtkToolItem*)_obj ) );
}


/*#
    @method get_toolbar_style GtkToolItem
    @brief Returns the toolbar style used for tool_item.
    @return A GtkToolbarStyle indicating the toolbar style used for tool_item.

    Custom subclasses of GtkToolItem should call this function in the handler of
    the GtkToolItem::toolbar_reconfigured signal to find out in what style
    the toolbar is displayed and change themselves accordingly

    Possibilities are:

    - GTK_TOOLBAR_BOTH, meaning the tool item should show both an icon and a label, stacked vertically
    - GTK_TOOLBAR_ICONS, meaning the toolbar shows only icons
    - GTK_TOOLBAR_TEXT, meaning the tool item should only show text
    - GTK_TOOLBAR_BOTH_HORIZ, meaning the tool item should show both an icon and a
      label, arranged horizontally (however, note the "has_text_horizontally" that
      makes tool buttons not show labels when the toolbar style is GTK_TOOLBAR_BOTH_HORIZ.
 */
FALCON_FUNC ToolItem::get_toolbar_style( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_item_get_toolbar_style( (GtkToolItem*)_obj ) );
}


/*#
    @method get_relief_style GtkToolItem
    @brief Returns the relief style of tool_item.
    @return a GtkReliefStyle indicating the relief style used for tool_item.

    Custom subclasses of GtkToolItem should call this function in the handler
    of the "toolbar_reconfigured" signal to find out the relief style of buttons.
 */
FALCON_FUNC ToolItem::get_relief_style( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_item_get_relief_style( (GtkToolItem*)_obj ) );
}


#if GTK_CHECK_VERSION( 2, 20, 0 )
/*#
    @method get_text_alignment GtkToolItem
    @brief Returns the text alignment used for tool_item.
    @return a numeric value indicating the horizontal text alignment used for tool_item

    Custom subclasses of GtkToolItem should call this function to find out how
    text should be aligned.
 */
FALCON_FUNC ToolItem::get_text_alignment( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_tool_item_get_text_alignment( (GtkToolItem*)_obj ) );
}


/*#
    @method get_text_orientation GtkToolItem
    @brief Returns the text orientation used for tool_item.
    @return a GtkOrientation indicating the text orientation used for tool_item

    Custom subclasses of GtkToolItem should call this function to find out how
    text should be orientated.
 */
FALCON_FUNC ToolItem::get_text_orientation( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_item_get_text_orientation( (GtkToolItem*)_obj ) );
}
#endif


/*#
    @method retrieve_proxy_menu_item GtkToolItem
    @brief Returns the GtkMenuItem that was last set by gtk_tool_item_set_proxy_menu_item(), ie. the GtkMenuItem that is going to appear in the overflow menu.
    @return The GtkMenuItem that is going to appear in the overflow menu for tool_item.
 */
//FALCON_FUNC ToolItem::retrieve_proxy_menu_item( VMARG );


//FALCON_FUNC ToolItem::get_proxy_menu_item( VMARG )



/*#
    @method set_proxy_menu_item GtkToolItem
    @brief Sets the GtkMenuItem used in the toolbar overflow menu.
    @param menu_item_id a string used to identify menu_item
    @param menu_item a GtkMenuItem to be used in the overflow menu

    The menu_item_id is used to identify the caller of this function and should
    also be used with gtk_tool_item_get_proxy_menu_item().
 */
FALCON_FUNC ToolItem::set_proxy_menu_item( VMARG )
{
    const char* spec = "S,GtkMenuItem";
    Gtk::ArgCheck1 args( vm, spec );
    const gchar* menu_item_id = args.getCString( 0 );
    CoreGObject* o_menu_item = args.getCoreGObject( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_menu_item, GtkMenuItem ) )
        throw_inv_params( spec );
#endif
    GtkWidget* menu_item = (GtkWidget*) o_menu_item->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_set_proxy_menu_item( (GtkToolItem*)_obj, menu_item_id, menu_item );
}


/*#
    @method rebuild_menu GtkToolItem
    @brief Calling this function signals to the toolbar that the overflow menu item for tool_item has changed.

    If the overflow menu is visible when this function it called, the menu will
    be rebuilt.

    The function must be called when the tool item changes what it will do in
    response to the "create-menu-proxy" signal.
 */
FALCON_FUNC ToolItem::rebuild_menu( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_rebuild_menu( (GtkToolItem*)_obj );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method toolbar_reconfigured GtkToolItem
    @brief Emits the signal "toolbar_reconfigured" on tool_item.

    GtkToolbar and other GtkToolShell implementations use this function to notify
    children, when some aspect of their configuration changes.
 */
FALCON_FUNC ToolItem::toolbar_reconfigured( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_toolbar_reconfigured( (GtkToolItem*)_obj );
}
#endif


//FALCON_FUNC ToolItem::get_text_size_group( VMARG );


} // Gtk
} // Falcon
