/**
 *  \file gtk_Container.cpp
 */

#include "gtk_Container.hpp"

#include "gtk_Widget.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {


void Container::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Container = mod->addClass( "GtkContainer", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkWidget" ) );
    c_Container->getClassDef()->addInheritance( in );

    c_Container->getClassDef()->factory( &Container::factory );

    Gtk::MethodTab methods[] =
    {
    { "add",                    &Container::add },
    { "remove",                 &Container::remove },
    //{ "add_with_properties",  &Container::add_with_properties },
    { "get_resize_mode",        &Container::get_resize_mode },
    { "set_resize_mode",        &Container::set_resize_mode },
    { "check_resize",           &Container::check_resize },
    //{ "foreach",              &Container::foreach },
    //{ "foreach_full",         &Container::foreach_full },
    //{ "get_children",         &Container::get_children },
    { "set_reallocate_redraws", &Container::set_reallocate_redraws },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_focus_child",        &Container::get_focus_child },
#endif
    { "set_focus_child",        &Container::set_focus_child },
    //{ "get_focus_vadjustment",&Container::get_focus_vadjustment },
    //{ "set_focus_vadjustment",&Container::set_focus_vadjustment },
    //{ "get_focus_hadjustment",&Container::get_focus_hadjustment },
    //{ "set_focus_hadjustment",&Container::set_focus_hadjustment },
    { "resize_children",        &Container::resize_children },
    { "child_type",             &Container::child_type },
    //{ "child_get",            &Container::child_get },
    //{ "child_set",            &Container::child_set },
    //{ "child_get_property",   &Container::child_get_property },
    //{ "child_set_property",   &Container::child_set_property },
    //{ "child_get_valist",     &Container::child_get_valist },
    //{ "child_set_valist",     &Container::child_set_valist },
    //{ "forall",               &Container::forall },
    { "get_border_width",       &Container::get_border_width },
    { "set_border_width",       &Container::set_border_width },
    //{ "propagate_expose",     &Container::propagate_expose },
    //{ "get_focus_chain",      &Container::get_focus_chain },
    //{ "unset_focus_chain",    &Container::unset_focus_chain },
    //{ "class_find_child_property",&Container::class_find_child_property },
    //{ "class_install_child_property",&Container::class_install_child_property },
    //{ "class_list_child_properties",&Container::class_list_child_properties },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Container, meth->name, meth->cb );

}


Container::Container( const Falcon::CoreClass* gen, const GtkContainer* ctn )
    :
    Gtk::CoreGObject( gen, (GObject*) ctn )
{}


Falcon::CoreObject* Container::factory( const Falcon::CoreClass* gen, void* ctn, bool )
{
    return new Container( gen, (GtkContainer*) ctn );
}


/*#
    @class GtkContainer
    @brief Abstract container class.

    This is the abstract container from which all gtk+ widgets which hold other
    items derive from. It mainly houses virtual functions used for inserting
    and removing children. Containers in gtk+ may hold one item or many items
    depending on the implementation.
 */

/*#
    @method add GtkContainer
    @brief Add a widget to the container.
    @param widget The widget
 */
FALCON_FUNC Container::add( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() ||
        !IS_DERIVED( i_wdt, GtkWidget ) )
    {
        throw_inv_params( "GtkWidget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    gtk_container_add( (GtkContainer*)_obj, wdt );
}


/*#
    @method remove GtkContainer
    @brief Removes widget from container. Widget must be inside container.
    @param widget
 */
FALCON_FUNC Container::remove( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_container_remove( (GtkContainer*)_obj, wdt );
}


//FALCON_FUNC Container::add_with_properties( VMARG );


/*#
    @method get_resize_mode GtkContainer
    @brief Returns the resize mode for the container.
 */
FALCON_FUNC Container::get_resize_mode( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_container_get_resize_mode( (GtkContainer*)_obj ) );
}


/*#
    @method set_resize_mode GtkContainer
    @brief Sets the resize mode for the container.

    The resize mode of a container determines whether a resize request will be passed
    to the container's parent, queued for later execution or executed immediately.
 */
FALCON_FUNC Container::set_resize_mode( VMARG )
{
    Item* i_mode = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mode || i_mode->isNil() || !i_mode->isInteger() )
        throw_inv_params( "I" );
#endif
    int mode = i_mode->asInteger();
#ifndef NO_PARAMETER_CHECK
    if ( mode < 0 || mode > 2 )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_container_set_resize_mode( (GtkContainer*)_obj, (GtkResizeMode)mode );
}


/*#
    @method check_resize GtkContainer
    @brief ???
 */
FALCON_FUNC Container::check_resize( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_container_check_resize( (GtkContainer*)_obj );
}


//FALCON_FUNC Container::foreach( VMARG );

//FALCON_FUNC Container::foreach_full( VMARG );

//FALCON_FUNC Container::get_children( VMARG );


/*#
    @method set_reallocate_redraws GtkContainer
    @brief Sets the reallocate_redraws flag of the container to the given value.
    @param need_redraws (boolean)

    Containers requesting reallocation redraws get automatically redrawn if any
    of their children changed allocation.
 */
FALCON_FUNC Container::set_reallocate_redraws( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_container_set_reallocate_redraws( (GtkContainer*)_obj,
        i_bool->asBoolean() ? TRUE : FALSE );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_focus_child GtkContainer
    @brief Returns the current focus child widget inside container.
 */
FALCON_FUNC Container::get_focus_child( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_container_get_focus_child( (GtkContainer*)_obj );
    if ( wdt )
    {
        Item* wki = vm->findWKI( "GtkWidget" );
        vm->retval( new Gtk::Widget( wki->asClass(), wdt ) );
    }
    else
        vm->retnil();
}
#endif


/*#
    @method set_focus_child GtkContainer
    @brief Sets, or unsets if child is NULL, the focused child of container.
    @param widget
 */
FALCON_FUNC Container::set_focus_child( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_container_set_focus_child( (GtkContainer*)_obj, wdt );
}


//FALCON_FUNC Container::get_focus_vadjustment( VMARG );

//FALCON_FUNC Container::set_focus_vadjustment( VMARG );

//FALCON_FUNC Container::get_focus_hadjustment( VMARG );

//FALCON_FUNC Container::set_focus_hadjustment( VMARG );


/*#
    @method resize_children GtkContainer
    @brief ???
 */
FALCON_FUNC Container::resize_children( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_container_resize_children( (GtkContainer*)_obj );
}


/*#
    @method child_type GtkContainer
    @brief Returns the type of the children supported by the container.
    @return type identifier (integer)

    Note that this may return G_TYPE_NONE to indicate that no more children can
    be added, e.g. for a GtkPaned which already has two children.
 */
FALCON_FUNC Container::child_type( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int) gtk_container_child_type( (GtkContainer*)_obj ) );
}


//FALCON_FUNC Container::child_get( VMARG );

//FALCON_FUNC Container::child_set( VMARG );

//FALCON_FUNC Container::child_get_property( VMARG );

//FALCON_FUNC Container::child_set_property( VMARG );

//FALCON_FUNC Container::child_get_valist( VMARG );

//FALCON_FUNC Container::child_set_valist( VMARG );

//FALCON_FUNC Container::forall( VMARG );


/*#
    @method get_border_width GtkContainer
    @brief Retrieves the border width of the container.
    @return (integer)
 */
FALCON_FUNC Container::get_border_width( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int) gtk_container_get_border_width( (GtkContainer*)_obj ) );
}


/*#
    @method set_border_width GtkContainer
    @brief Sets the border width of the container.

    The border width of a container is the amount of space to leave around the outside
    of the container. The only exception to this is GtkWindow; because toplevel
    windows can't leave space outside, they leave the space inside.
    The border is added on all sides of the container. To add space to only one
    side, one approach is to create a GtkAlignment widget, call set_size_request()
    to give it a size, and place it on the side of the container as a spacer.
 */
FALCON_FUNC Container::set_border_width( VMARG )
{
    Item* i_w = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || i_w->isNil() || !i_w->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_container_set_border_width( (GtkContainer*)_obj, i_w->asInteger() );
}


//FALCON_FUNC Container::propagate_expose( VMARG );

//FALCON_FUNC Container::get_focus_chain( VMARG );

//FALCON_FUNC Container::unset_focus_chain( VMARG );

//FALCON_FUNC Container::class_find_child_property( VMARG );

//FALCON_FUNC Container::class_install_child_property( VMARG );

//FALCON_FUNC Container::class_list_child_properties( VMARG );


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
