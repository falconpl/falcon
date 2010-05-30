/**
 *  \file gtk_Box.cpp
 */

#include "gtk_Box.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Box::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Box = mod->addClass( "GtkBox", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_Box->getClassDef()->addInheritance( in );

    //c_Box->setWKS( true );
    //c_Box->getClassDef()->factory( &Box::factory );

    Gtk::MethodTab methods[] =
    {
    { "pack_start",             &Box::pack_start },
    { "pack_end",               &Box::pack_end },
    { "pack_start_defaults",    &Box::pack_start_defaults },
    { "pack_end_defaults",      &Box::pack_end_defaults },
    { "get_homogeneous",        &Box::get_homogeneous },
    { "set_homogeneous",        &Box::set_homogeneous },
    { "get_spacing",            &Box::get_spacing },
    { "reorder_child",          &Box::reorder_child },
    { "query_child_packing",    &Box::query_child_packing },
    { "set_child_packing",      &Box::set_child_packing },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Box, meth->name, meth->cb );
}


Box::Box( const Falcon::CoreClass* gen, const GtkBox* box )
    :
    Gtk::CoreGObject( gen, (GObject*) box )
{}


Falcon::CoreObject* Box::factory( const Falcon::CoreClass* gen, void* box, bool )
{
    return new Box( gen, (GtkBox*) box );
}


/*#
    @class GtkBox
    @brief Base class for box containers

    GtkBox is an abstract widget which encapsulates functionality for a particular
    kind of container, one that organizes a variable number of widgets into a
    rectangular area. GtkBox has a number of derived classes, e.g. GtkHBox and GtkVBox.

    The rectangular area of a GtkBox is organized into either a single row or a
    single column of child widgets depending upon whether the box is of type
    GtkHBox or GtkVBox, respectively. Thus, all children of a GtkBox are allocated
    one dimension in common, which is the height of a row, or the width of a column.

    GtkBox uses a notion of packing. Packing refers to adding widgets with
    reference to a particular position in a GtkContainer. For a GtkBox, there
    are two reference positions: the start and the end of the box. For a GtkVBox,
    the start is defined as the top of the box and the end is defined as the bottom.
    For a GtkHBox the start is defined as the left side and the end is defined
    as the right side.

    Use repeated calls to gtk_box_pack_start() to pack widgets into a GtkBox
    from start to end. Use gtk_box_pack_end() to add widgets from end to start.
    You may intersperse these calls and add widgets from both ends of the same GtkBox.

    Use gtk_box_pack_start_defaults() or gtk_box_pack_end_defaults() to pack
    widgets into a GtkBox if you do not need to specify the "expand", "fill",
    or "padding" child properties for the child to be added.

    Because GtkBox is a GtkContainer, you may also use gtk_container_add() to
    insert widgets into the box, and they will be packed as if with
    gtk_box_pack_start_defaults(). Use gtk_container_remove() to remove widgets
    from the GtkBox.

    Use gtk_box_set_homogeneous() to specify whether or not all children of the
    GtkBox are forced to get the same amount of space.

    Use gtk_box_set_spacing() to determine how much space will be minimally
    placed between all children in the GtkBox.

    Use gtk_box_reorder_child() to move a GtkBox child to a different place in the box.

    Use gtk_box_set_child_packing() to reset the "expand", "fill" and "padding"
    child properties. Use gtk_box_query_child_packing() to query these fields.

    [...]
 */


/*#
    @method pack_start GtkBox
    @brief Adds child to box, packed with reference to the start of box.
    @param child the GtkWidget to be added to box
    @param expand TRUE if the new child is to be given extra space allocated to box. The extra space will be divided evenly between all children of box that use this option.
    @param fill TRUE if space given to child by the expand option is actually allocated to child, rather than just padding it. This parameter has no effect if expand is set to FALSE. A child is always allocated the full height of a GtkHBox and the full width of a GtkVBox. This option affects the other dimension.
    @param padding extra space in pixels to put between this child and its neighbors, over and above the global amount specified by "spacing" property. If child is a widget at one of the reference ends of box, then padding pixels are also put between child and the reference edge of box.

    The child is packed after any other child packed with reference to the start of box.
 */
FALCON_FUNC Box::pack_start( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_expand = vm->param( 1 );
    Item* i_fill = vm->param( 2 );
    Item* i_padding = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || !i_expand || !i_fill || !i_padding
        || !i_child->isObject()
        || !IS_DERIVED( i_child, GtkWidget )
        || !i_expand->isBoolean()
        || !i_fill->isBoolean()
        || !i_padding->isInteger() )
        throw_inv_params( "GtkWidget,B,B,I" );
#endif
    int padding = i_padding->asInteger();
#ifndef NO_PARAMETER_CHECK
    if ( padding < 0 )
        throw_inv_params( "GtkWidget,B,B,I" );
#endif
    GtkWidget* child = (GtkWidget*) COREGOBJECT( i_child )->getGObject();

    MYSELF;
    GET_OBJ( self );
    gtk_box_pack_start( (GtkBox*)_obj, child, (gboolean) i_expand->asBoolean(),
                        (gboolean) i_fill->asBoolean(), padding );
}


/*#
    @method pack_end GtkBox
    @brief Adds child to box, packed with reference to the end of box.
    @param child the GtkWidget to be added to box
    @param expand TRUE if the new child is to be given extra space allocated to box. The extra space will be divided evenly between all children of box that use this option.
    @param fill TRUE if space given to child by the expand option is actually allocated to child, rather than just padding it. This parameter has no effect if expand is set to FALSE. A child is always allocated the full height of a GtkHBox and the full width of a GtkVBox. This option affects the other dimension.
    @param padding extra space in pixels to put between this child and its neighbors, over and above the global amount specified by "spacing" property. If child is a widget at one of the reference ends of box, then padding pixels are also put between child and the reference edge of box.

    The child is packed after (away from end of) any other child packed with
    reference to the end of box.
 */
FALCON_FUNC Box::pack_end( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_expand = vm->param( 1 );
    Item* i_fill = vm->param( 2 );
    Item* i_padding = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || !i_expand || !i_fill || !i_padding
        || !i_child->isObject()
        || !IS_DERIVED( i_child, GtkWidget )
        || !i_expand->isBoolean()
        || !i_fill->isBoolean()
        || !i_padding->isInteger() )
        throw_inv_params( "GtkWidget,B,B,I" );
#endif
    int padding = i_padding->asInteger();
#ifndef NO_PARAMETER_CHECK
    if ( padding < 0 )
        throw_inv_params( "GtkWidget,B,B,I" );
#endif
    GtkWidget* child = (GtkWidget*) COREGOBJECT( i_child )->getGObject();

    MYSELF;
    GET_OBJ( self );
    gtk_box_pack_end( (GtkBox*)_obj, child, (gboolean) i_expand->asBoolean(),
                      (gboolean) i_fill->asBoolean(), padding );
}


/*#
    @method pack_start_defaults GtkBox
    @brief Adds widget to box, packed with reference to the start of box.
    @param widget the GtkWidget to be added to box

    The child is packed after any other child packed with reference to the start of box.

    Parameters for how to pack the child widget, "expand", "fill" and "padding",
    are given their default values, TRUE, TRUE, and 0, respectively.
 */
FALCON_FUNC Box::pack_start_defaults( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_box_pack_start_defaults( (GtkBox*)_obj, wdt );
}


/*#
    @method pack_end_defaults GtkBox
    @brief Adds widget to box, packed with reference to the end of box.
    @param widget the GtkWidget to be added to box

    The child is packed after any other child packed with reference to the start of box.

    Parameters for how to pack the child widget, "expand", "fill" and "padding",
    are given their default values, TRUE, TRUE, and 0, respectively.
 */
FALCON_FUNC Box::pack_end_defaults( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_box_pack_end_defaults( (GtkBox*)_obj, wdt );
}


/*#
    @method get_homogeneous GtkBox
    @brief Returns whether the box is homogeneous (all children are the same size).
    @return TRUE if the box is homogeneous.
 */
FALCON_FUNC Box::get_homogeneous( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_box_get_homogeneous( (GtkBox*)_obj ) );
}


/*#
    @method set_homogeneous GtkBox
    @brief Sets the "homogeneous" property of box, controlling whether or not all children of box are given equal space in the box.
    @param homogeneous a boolean value, TRUE to create equal allotments, FALSE for variable allotments.
 */
FALCON_FUNC Box::set_homogeneous( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_box_set_homogeneous( (GtkBox*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_spacing GtkBox
    @brief Gets the value set by gtk_box_set_spacing().
    @return spacing between children
 */
FALCON_FUNC Box::get_spacing( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_box_get_spacing( (GtkBox*)_obj ) );
}


/*#
    @method reorder_child GtkBox
    @brief Moves child to a new position in the list of box children.
    @param child the GtkWidget to move
    @param position the new position for child in the list of children of box, starting from 0. If negative, indicates the end of the list

    The list is the children field of GtkBox, and contains both widgets packed
    GTK_PACK_START as well as widgets packed GTK_PACK_END, in the order that
    these widgets were added to box.

    A widget's position in the box children list determines where the widget is
    packed into box. A child widget at some position in the list will be packed
    just after all other widgets of the same packing type that appear earlier
    in the list.
 */
FALCON_FUNC Box::reorder_child( VMARG )
{
    Item* i_wdt = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget )
        || !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "GtkWidget,I" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_box_reorder_child( (GtkBox*)_obj, wdt, i_pos->asInteger() );
}


/*#
    @method query_child_packing GtkBox
    @brief Obtains information about how child is packed into box.
    @param child the GtkWidget of the child to query
    @return an array [ expand, fill, padding, GtkPackType ]
 */
FALCON_FUNC Box::query_child_packing( VMARG )
{
    Item* i_child = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || !( i_child->isObject() && IS_DERIVED( i_child, GtkWidget ) ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* child = (GtkWidget*) COREGOBJECT( i_child )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gboolean expand, fill;
    guint padding;
    GtkPackType ptype;
    gtk_box_query_child_packing( (GtkBox*)_obj,
                                 child, &expand, &fill, &padding, &ptype );
    CoreArray* arr = new CoreArray( 4 );
    arr->append( (bool) expand );
    arr->append( (bool) fill );
    arr->append( padding );
    arr->append( (int64) ptype );
    vm->retval( arr );
}


/*#
    @method set_child_packing GtkBox
    @brief Sets the way child is packed into box.
    @param child the GtkWidget of the child to set
    @param expand the new value of the "expand" child property
    @param fill the new value of the "fill" child property
    @param padding the new value of the "padding" child property
    @param pack_type the new value of the "pack-type" child property
 */
FALCON_FUNC Box::set_child_packing( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_expand = vm->param( 1 );
    Item* i_fill = vm->param( 2 );
    Item* i_padding = vm->param( 3 );
    Item* i_ptype = vm->param( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || !( i_child->isObject() && IS_DERIVED( i_child, GtkWidget ) )
        || !i_expand || !i_expand->isBoolean()
        || !i_fill || !i_fill->isBoolean()
        || !i_padding || !i_padding->isInteger()
        || !i_ptype || !i_ptype->isInteger() )
        throw_inv_params( "GtkWidget,B,B,I,GtkPackType" );
#endif
    GtkWidget* child = (GtkWidget*) COREGOBJECT( i_child )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_box_set_child_packing( (GtkBox*)_obj, child, (gboolean) i_expand->asBoolean(),
                               (gboolean) i_fill->asBoolean(), i_padding->asInteger(),
                               (GtkPackType) i_ptype->asInteger() );
}


} // Gtk
} // Falcon
