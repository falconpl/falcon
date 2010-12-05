/**
 *  \file gtk_ToolItemGroup.cpp
 */

#include "gtk_ToolItemGroup.hpp"

#if GTK_CHECK_VERSION( 2, 20, 0 )

#include "gtk_ToolItem.hpp"
#include "gtk_ToolShell.hpp"
#include "gtk_Widget.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ToolItemGroup::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ToolItemGroup = mod->addClass( "GtkToolItemGroup", &ToolItemGroup::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_ToolItemGroup->getClassDef()->addInheritance( in );

    c_ToolItemGroup->setWKS( true );
    c_ToolItemGroup->getClassDef()->factory( &ToolItemGroup::factory );

    Gtk::MethodTab methods[] =
    {
    { "get_collapsed",      &ToolItemGroup::get_collapsed },
    { "get_drop_item",      &ToolItemGroup::get_drop_item },
    { "get_ellipsize",      &ToolItemGroup::get_ellipsize },
    { "get_item_position",  &ToolItemGroup::get_item_position },
    { "get_n_items",        &ToolItemGroup::get_n_items },
    { "get_label",          &ToolItemGroup::get_label },
    { "get_label_widget",   &ToolItemGroup::get_label_widget },
    { "get_nth_item",       &ToolItemGroup::get_nth_item },
    { "get_header_relief",  &ToolItemGroup::get_header_relief },
    { "insert",             &ToolItemGroup::insert },
    { "set_collapsed",      &ToolItemGroup::set_collapsed },
    { "set_ellipsize",      &ToolItemGroup::set_ellipsize },
    { "set_item_position",  &ToolItemGroup::set_item_position },
    { "set_label",          &ToolItemGroup::set_label },
    { "set_label_widget",   &ToolItemGroup::set_label_widget },
    { "set_header_relief",  &ToolItemGroup::set_header_relief },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ToolItemGroup, meth->name, meth->cb );

    Gtk::ToolShell::clsInit( mod, c_ToolItemGroup );
}


ToolItemGroup::ToolItemGroup( const Falcon::CoreClass* gen, const GtkToolItemGroup* grp )
    :
    Gtk::CoreGObject( gen, (GObject*) grp )
{}


Falcon::CoreObject* ToolItemGroup::factory( const Falcon::CoreClass* gen, void* grp, bool )
{
    return new ToolItemGroup( gen, (GtkToolItemGroup*) grp );
}


/*#
    @class GtkToolItemGroup
    @brief A sub container used in a tool palette
    @param label the label of the new group

    A GtkToolItemGroup is used together with GtkToolPalette to add GtkToolItems
    to a palette like container with different categories and drag and drop support.
 */
FALCON_FUNC ToolItemGroup::init( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* lbl = args.getCString( 0 );
    MYSELF;
    self->setObject( (GObject*) gtk_tool_item_group_new( lbl ) );
}


/*#
    @method get_collapsed GtkToolItemGroup
    @brief Gets whether group is collapsed or expanded.
    @return TRUE if group is collapsed, FALSE if it is expanded
 */
FALCON_FUNC ToolItemGroup::get_collapsed( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_item_group_get_collapsed( (GtkToolItemGroup*)_obj ) );
}


/*#
    @method get_drop_item GtkToolItemGroup
    @brief Gets the tool item at position (x, y).
    @param x the x position
    @param y the y position
    @return the GtkToolItem at position (x, y)
 */
FALCON_FUNC ToolItemGroup::get_drop_item( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkToolItem* itm = gtk_tool_item_group_get_drop_item(
                    (GtkToolItemGroup*)_obj, i_x->asInteger(), i_y->asInteger() );
    vm->retval( new Gtk::ToolItem( vm->findWKI( "GtkToolItem" )->asClass(), itm ) );
}


/*#
    @method get_ellipsize GtkToolItemGroup
    @brief Gets the ellipsization mode of group.
    @return the PangoEllipsizeMode of group
 */
FALCON_FUNC ToolItemGroup::get_ellipsize( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_item_group_get_ellipsize( (GtkToolItemGroup*)_obj ) );
}


/*#
    @method get_item_position GtkToolItemGroup
    @brief Gets the position of item in group as index.
    @param item a GtkToolItem
    @return the index of item in group or -1 if item is no child of group
 */
FALCON_FUNC ToolItemGroup::get_item_position( VMARG )
{
    Item* i_itm = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_itm || !i_itm->isObject() || !IS_DERIVED( i_itm, GtkToolItem ) )
        throw_inv_params( "GtkToolItem" );
#endif
    GtkToolItem* itm = (GtkToolItem*) COREGOBJECT( i_itm )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_tool_item_group_get_item_position( (GtkToolItemGroup*)_obj, itm ) );
}


/*#
    @method get_n_items GtkToolItemGroup
    @brief Gets the number of tool items in group.
    @return the number of tool items in group
 */
FALCON_FUNC ToolItemGroup::get_n_items( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_item_group_get_n_items( (GtkToolItemGroup*)_obj ) );
}


/*#
    @method get_label GtkToolItemGroup
    @brief Gets the label of group.
    @return the label of group.

    Note that NULL is returned if a custom label has been set with
    gtk_tool_item_group_set_label_widget().

 */
FALCON_FUNC ToolItemGroup::get_label( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_tool_item_group_get_label( (GtkToolItemGroup*)_obj );
    if ( lbl )
        vm->retval( UTF8String( lbl ) );
    else
        vm->retnil();
}


/*#
    @method get_label_widget GtkToolItemGroup
    @brief Gets the label widget of group.
    @return the label widget of group
 */
FALCON_FUNC ToolItemGroup::get_label_widget( VMARG )
{ // return GtkLabel instead?
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* lbl = gtk_tool_item_group_get_label_widget( (GtkToolItemGroup*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), lbl ) );
}


/*#
    @method get_nth_item GtkToolItemGroup
    @brief Gets the tool item at index in group.
    @param index the index
    @return the GtkToolItem at index
 */
FALCON_FUNC ToolItemGroup::get_nth_item( VMARG )
{
    Item* i_idx = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_idx || !i_idx->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkToolItem* itm = gtk_tool_item_group_get_nth_item( (GtkToolItemGroup*)_obj, i_idx->asInteger() );
    vm->retval( new Gtk::ToolItem( vm->findWKI( "GtkToolItem" )->asClass(), itm ) );
}


/*#
    @method get_header_relief GtkToolItemGroup
    @brief Gets the relief mode of the header button of group.
    @return the GtkReliefStyle
 */
FALCON_FUNC ToolItemGroup::get_header_relief( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_item_group_get_header_relief( (GtkToolItemGroup*)_obj ) );
}


/*#
    @method insert GtkToolItemGroup
    @brief Inserts item at position in the list of children of group.
    @param item the GtkToolItem to insert into group
    @param position the position of item in group, starting with 0. The position -1 means end of list.
 */
FALCON_FUNC ToolItemGroup::insert( VMARG )
{
    Item* i_itm = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_itm || !i_itm->isObject() || !IS_DERIVED( i_itm, GtkToolItem )
        || !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkToolItem,I" );
#endif
    GtkToolItem* itm = (GtkToolItem*) COREGOBJECT( i_itm )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_group_insert( (GtkToolItemGroup*)_obj, itm, i_pos->asInteger() );
}


/*#
    @method set_collapsed GtkToolItemGroup
    @brief Sets whether the group should be collapsed or expanded.
    @param collapsed whether the group should be collapsed or expanded
 */
FALCON_FUNC ToolItemGroup::set_collapsed( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_group_set_collapsed( (GtkToolItemGroup*)_obj,
                                       i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_ellipsize GtkToolItemGroup
    @brief Sets the ellipsization mode which should be used by labels in group.
    @param ellipsize the PangoEllipsizeMode labels in group should use
 */
FALCON_FUNC ToolItemGroup::set_ellipsize( VMARG )
{
    Item* i_mode = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mode || !i_mode->isInteger() )
        throw_inv_params( "PangoEllipsizeMode" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_group_set_ellipsize( (GtkToolItemGroup*)_obj,
                                       (PangoEllipsizeMode) i_mode->asInteger() );
}


/*#
    @method set_item_position GtkToolItemGroup
    @brief Sets the position of item in the list of children of group.
    @param item the GtkToolItem to move to a new position, should be a child of group.
    @param position the new position of item in group, starting with 0. The position -1 means end of list.
 */
FALCON_FUNC ToolItemGroup::set_item_position( VMARG )
{
    Item* i_itm = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_itm || !i_itm->isObject() || !IS_DERIVED( i_itm, GtkToolItem )
        || !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkToolItem,I" );
#endif
    GtkToolItem* itm = (GtkToolItem*) COREGOBJECT( i_itm )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_group_set_item_position( (GtkToolItemGroup*)_obj, itm, i_pos->asInteger() );
}


/*#
    @method set_label GtkToolItemGroup
    @brief Sets the label of the tool item group.
    @param label the new human-readable label of of the group

    The label is displayed in the header of the group.
 */
FALCON_FUNC ToolItemGroup::set_label( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* lbl = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_group_set_label( (GtkToolItemGroup*)_obj, lbl );
}


/*#
    @method set_label_widget GtkToolItemGroup
    @brief Sets the label of the tool item group.
    @param label_widget the widget to be displayed in place of the usual label

    The label widget is displayed in the header of the group, in place of the usual label.
 */
FALCON_FUNC ToolItemGroup::set_label_widget( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !i_wdt->isObject() || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_group_set_label_widget( (GtkToolItemGroup*)_obj, wdt );
}


/*#
    @method set_header_relief GtkToolItemGroup
    @brief Set the button relief of the group header.
    @param style the GtkReliefStyle
 */
FALCON_FUNC ToolItemGroup::set_header_relief( VMARG )
{
    Item* i_styl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_styl || !i_styl->isInteger() )
        throw_inv_params( "GtkReliefStyle" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_item_group_set_header_relief( (GtkToolItemGroup*)_obj,
                                           (GtkReliefStyle) i_styl->asInteger() );
}


} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 20, 0 )
