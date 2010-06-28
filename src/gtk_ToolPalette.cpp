/**
 *  \file gtk_ToolPalette.cpp
 */

#include "gtk_ToolPalette.hpp"

#if GTK_CHECK_VERSION( 2, 20, 0 )

#include "gtk_Adjustment.hpp"
#include "gtk_Buildable.hpp"
#include "gtk_Orientable.hpp"
#include "gtk_ToolItem.hpp"
#include "gtk_ToolItemGroup.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ToolPalette::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ToolPalette = mod->addClass( "GtkToolPalette", &ToolPalette::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_ToolPalette->getClassDef()->addInheritance( in );

    c_ToolPalette->setWKS( true );
    c_ToolPalette->getClassDef()->factory( &ToolPalette::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_set_scroll_adjustments",&ToolPalette::signal_set_scroll_adjustments },
    { "get_exclusive",          &ToolPalette::get_exclusive },
    { "set_exclusive",          &ToolPalette::set_exclusive },
    { "get_expand",             &ToolPalette::get_expand },
    { "set_expand",             &ToolPalette::set_expand },
    { "get_group_position",     &ToolPalette::get_group_position },
    { "set_group_position",     &ToolPalette::set_group_position },
    { "get_icon_size",          &ToolPalette::get_icon_size },
    { "set_icon_size",          &ToolPalette::set_icon_size },
    { "unset_icon_size",        &ToolPalette::unset_icon_size },
    { "get_style",              &ToolPalette::get_style },
    { "set_style",              &ToolPalette::set_style },
    { "unset_style",            &ToolPalette::unset_style },
#if 0 // todo
    { "add_drag_dest",          &ToolPalette::add_drag_dest },
    { "get_drag_item",          &ToolPalette::get_drag_item },
    { "get_drag_target_group",  &ToolPalette::get_drag_target_group },
    { "get_drag_target_item",   &ToolPalette::get_drag_target_item },
#endif
    { "get_drop_group",         &ToolPalette::get_drop_group },
    { "get_drop_item",          &ToolPalette::get_drop_item },
    { "set_drag_source",        &ToolPalette::set_drag_source },
    { "get_hadjustment",        &ToolPalette::get_hadjustment },
    { "get_vadjustment",        &ToolPalette::get_vadjustment },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ToolPalette, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_ToolPalette );
    Gtk::Orientable::clsInit( mod, c_ToolPalette );
}


ToolPalette::ToolPalette( const Falcon::CoreClass* gen, const GtkToolPalette* itm )
    :
    Gtk::CoreGObject( gen, (GObject*) itm )
{}


Falcon::CoreObject* ToolPalette::factory( const Falcon::CoreClass* gen, void* itm, bool )
{
    return new ToolPalette( gen, (GtkToolPalette*) itm );
}


/*#
    @class GtkToolPalette
    @brief A tool palette with categories

    A GtkToolPalette allows you to add GtkToolItems to a palette-like container
    with different categories and drag and drop support.

    [...]
 */
FALCON_FUNC ToolPalette::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_tool_palette_new() );
}


/*#
    @method signal_set_scroll_adjustments GtkToolPalette
    @brief Set the scroll adjustments for the viewport.

    Usually scrolled containers like GtkScrolledWindow will emit this signal to
    connect two instances of GtkScrollbar to the scroll directions of the GtkToolpalette.
 */
FALCON_FUNC ToolPalette::signal_set_scroll_adjustments( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "set_scroll_adjustments",
                             (void*) &ToolPalette::on_set_scroll_adjustments, vm );
}


void ToolPalette::on_set_scroll_adjustments( GtkToolPalette* obj, GtkAdjustment* hadj,
                                             GtkAdjustment* vadj, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "set_scroll_adjustments", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkAdjustment" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_set_scroll_adjustments", it ) )
            {
                printf(
                "[GtkToolPalette::on_set_scroll_adjustments] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::Adjustment( wki->asClass(), hadj ) );
        vm->pushParam( new Gtk::Adjustment( wki->asClass(), vadj ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method get_exclusive GtkToolPalette
    @brief Gets whether group is exclusive or not.
    @param group a GtkToolItemGroup which is a child of palette
    @return TRUE if group is exclusive
 */
FALCON_FUNC ToolPalette::get_exclusive( VMARG )
{
    Item* i_grp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isObject() && IS_DERIVED( i_grp, GtkToolItemGroup ) ) )
        throw_inv_params( "GtkToolItemGroup" );
#endif
    GtkToolItemGroup* grp = (GtkToolItemGroup*) COREGOBJECT( i_grp )->getGObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_palette_get_exclusive( (GtkToolPalette*)_obj, grp ) );
}


/*#
    @method set_exclusive GtkToolPalette
    @brief Sets whether the group should be exclusive or not.
    @param group a GtkToolItemGroup which is a child of palette
    @param exclusive whether the group should be exclusive or not

    If an exclusive group is expanded all other groups are collapsed.
 */
FALCON_FUNC ToolPalette::set_exclusive( VMARG )
{
    Item* i_grp = vm->param( 0 );
    Item* i_bool = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isObject() && IS_DERIVED( i_grp, GtkToolItemGroup ) )
        || !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "GtkToolItemGroup,B" );
#endif
    GtkToolItemGroup* grp = (GtkToolItemGroup*) COREGOBJECT( i_grp )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_palette_set_exclusive( (GtkToolPalette*)_obj, grp, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_expand GtkToolPalette
    @brief Gets whether group should be given extra space.
    @param group a GtkToolItemGroup which is a child of palette
    @return TRUE if group should be given extra space, FALSE otherwise
 */
FALCON_FUNC ToolPalette::get_expand( VMARG )
{
    Item* i_grp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isObject() && IS_DERIVED( i_grp, GtkToolItemGroup ) ) )
        throw_inv_params( "GtkToolItemGroup" );
#endif
    GtkToolItemGroup* grp = (GtkToolItemGroup*) COREGOBJECT( i_grp )->getGObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_palette_get_expand( (GtkToolPalette*)_obj, grp ) );
}


/*#
    @method set_expand GtkToolPalette
    @brief Sets whether the group should be given extra space.
    @param group a GtkToolItemGroup which is a child of palette
    @param expand whether the group should be given extra space
 */
FALCON_FUNC ToolPalette::set_expand( VMARG )
{
    Item* i_grp = vm->param( 0 );
    Item* i_bool = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isObject() && IS_DERIVED( i_grp, GtkToolItemGroup ) )
        || !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "GtkToolItemGroup,B" );
#endif
    GtkToolItemGroup* grp = (GtkToolItemGroup*) COREGOBJECT( i_grp )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_palette_set_expand( (GtkToolPalette*)_obj, grp, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_group_position GtkToolPalette
    @brief Gets the position of group in palette as index.
    @param group a GtkToolItemGroup
    @return the index of group or -1 if group is not a child of palette
 */
FALCON_FUNC ToolPalette::get_group_position( VMARG )
{
    Item* i_grp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isObject() && IS_DERIVED( i_grp, GtkToolItemGroup ) ) )
        throw_inv_params( "GtkToolItemGroup" );
#endif
    GtkToolItemGroup* grp = (GtkToolItemGroup*) COREGOBJECT( i_grp )->getGObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_tool_palette_get_group_position( (GtkToolPalette*)_obj, grp ) );
}


/*#
    @method set_group_position GtkToolPalette
    @brief Sets the position of the group as an index of the tool palette.
    @param group a GtkToolItemGroup which is a child of palette
    @param position a new index for group

    If position is 0 the group will become the first child, if position is -1
    it will become the last child.
 */
FALCON_FUNC ToolPalette::set_group_position( VMARG )
{
    Item* i_grp = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isObject() && IS_DERIVED( i_grp, GtkToolItemGroup ) )
        || !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkToolItemGroup,I" );
#endif
    GtkToolItemGroup* grp = (GtkToolItemGroup*) COREGOBJECT( i_grp )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_palette_set_group_position( (GtkToolPalette*)_obj, grp, i_pos->asInteger() );
}


/*#
    @method get_icon_size GtkToolPalette
    @brief Gets the size of icons in the tool palette.
    @return the GtkIconSize of icons in the tool palette (GtkIconSize).
 */
FALCON_FUNC ToolPalette::get_icon_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_palette_get_icon_size( (GtkToolPalette*)_obj ) );
}


/*#
    @method set_icon_size GtkToolPalette
    @brief Sets the size of icons in the tool palette.
    @param icon_size the GtkIconSize that icons in the tool palette shall have.
 */
FALCON_FUNC ToolPalette::set_icon_size( VMARG )
{
    Item* i_sz = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sz || !i_sz->isInteger() )
        throw_inv_params( "GtkIconSize" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_palette_set_icon_size( (GtkToolPalette*)_obj, (GtkIconSize) i_sz->asInteger() );
}


/*#
    @method unset_icon_size GtkToolPalette
    @brief Unsets the tool palette icon size set with gtk_tool_palette_set_icon_size(), so that user preferences will be used to determine the icon size.
 */
FALCON_FUNC ToolPalette::unset_icon_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_tool_palette_unset_icon_size( (GtkToolPalette*)_obj );
}


/*#
    @method get_style GtkToolPalette
    @brief Gets the style (icons, text or both) of items in the tool palette.
    @return the GtkToolbarStyle of items in the tool palette.
 */
FALCON_FUNC ToolPalette::get_style( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tool_palette_get_style( (GtkToolPalette*)_obj ) );
}


/*#
    @method set_style GtkToolPalette
    @brief Sets the style (text, icons or both) of items in the tool palette.
    @param style the GtkToolbarStyle that items in the tool palette shall have
 */
FALCON_FUNC ToolPalette::set_style( VMARG )
{
    Item* i_style = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_style || !i_style->isInteger() )
        throw_inv_params( "GtkToolbarStyle" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_palette_set_style( (GtkToolPalette*)_obj, (GtkToolbarStyle) i_style->asInteger() );
}


/*#
    @method unset_style GtkToolPalette
    @brief Unsets a toolbar style set with gtk_tool_palette_set_style(), so that user preferences will be used to determine the toolbar style.
 */
FALCON_FUNC ToolPalette::unset_style( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_tool_palette_unset_style( (GtkToolPalette*)_obj );
}


#if 0 // todo
FALCON_FUNC ToolPalette::add_drag_dest( VMARG );
FALCON_FUNC ToolPalette::get_drag_item( VMARG );
FALCON_FUNC ToolPalette::get_drag_target_group( VMARG );
FALCON_FUNC ToolPalette::get_drag_target_item( VMARG );
#endif


/*#
    @method get_drop_group GtkToolPalette
    @brief Gets the group at position (x, y).
    @param x the x position
    @param y the y position
    @return the GtkToolItemGroup at position or NULL  if there is no such group
 */
FALCON_FUNC ToolPalette::get_drop_group( VMARG )
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
    GtkToolItemGroup* grp = gtk_tool_palette_get_drop_group( (GtkToolPalette*)_obj,
                                                             i_x->asInteger(),
                                                             i_y->asInteger() );
    if ( grp )
        vm->retval( new Gtk::ToolItemGroup( vm->findWKI( "GtkToolItemGroup" )->asClass(), grp ) );
    else
        vm->retnil();
}


/*#
    @method get_drop_item GtkToolPalette
    @brief Gets the item at position (x, y).
    @param x the x position
    @param y the y position
    @return the GtkToolItem at position or NULL if there is no such item
 */
FALCON_FUNC ToolPalette::get_drop_item( VMARG )
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
    GtkToolItem* itm = gtk_tool_palette_get_drop_item( (GtkToolPalette*)_obj,
                                                       i_x->asInteger(),
                                                       i_y->asInteger() );
    if ( itm )
        vm->retval( new Gtk::ToolItem( vm->findWKI( "GtkToolItem" )->asClass(), itm ) );
    else
        vm->retnil();
}


/*#
    @method set_drag_source GtkToolPalette
    @brief Sets the tool palette as a drag source.
    @param targets the GtkToolPaletteDragTargets which the widget should support

    Enables all groups and items in the tool palette as drag sources on button 1
    and button 3 press with copy and move actions.
 */
FALCON_FUNC ToolPalette::set_drag_source( VMARG )
{
    Item* i_tgt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tgt || !i_tgt->isInteger() )
        throw_inv_params( "GtkToolPaletteDragTargets" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_palette_set_drag_source( (GtkToolPalette*)_obj,
                                      (GtkToolPaletteDragTargets) i_tgt->asInteger() );
}


/*#
    @method get_hadjustment GtkToolPalette
    @brief Gets the horizontal adjustment of the tool palette.
    @return the horizontal adjustment of palette (GtkAdjustment).
 */
FALCON_FUNC ToolPalette::get_hadjustment( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkAdjustment* adj = gtk_tool_palette_get_hadjustment( (GtkToolPalette*)_obj );
    vm->retval( new Gtk::Adjustment( vm->findWKI( "GtkAdjustment" )->asClass(), adj ) );
}


/*#
    @method get_vadjustment GtkToolPalette
    @brief Gets the vertical adjustment of the tool palette.
    @return the vertical adjustment of palette
 */
FALCON_FUNC ToolPalette::get_vadjustment( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkAdjustment* adj = gtk_tool_palette_get_vadjustment( (GtkToolPalette*)_obj );
    vm->retval( new Gtk::Adjustment( vm->findWKI( "GtkAdjustment" )->asClass(), adj ) );
}


} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 20, 0 )
