/**
 *  \file gtk_Box.cpp
 */

#include "gtk_Box.hpp"

#include <gtk/gtk.h>


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
    //{ "query_child_packing",    &Box::query_child_packing },
    //{ "set_child_packing",      &Box::set_child_packing },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Box, meth->name, meth->cb );
}


/*#
    @class GtkBox
    @brief Box class

    GtkBox is an abstract widget which encapsulates functionality for a particular
    kind of container, one that organizes a variable number of widgets into a
    rectangular area. GtkBox has a number of derived classes, e.g. GtkHBox and GtkVBox.
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
    GtkWidget* child = (GtkWidget*)((GData*)i_child->asObject()->getUserData())->obj();

    MYSELF;
    GET_OBJ( self );
    gtk_box_pack_start( (GtkBox*)_obj, child, i_expand->asBoolean() ? TRUE : FALSE,
        i_fill->asBoolean() ? TRUE : FALSE, padding );
}


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
    GtkWidget* child = (GtkWidget*)((GData*)i_child->asObject()->getUserData())->obj();

    MYSELF;
    GET_OBJ( self );
    gtk_box_pack_end( (GtkBox*)_obj, child, i_expand->asBoolean() ? TRUE : FALSE,
        i_fill->asBoolean() ? TRUE : FALSE, padding );
}


FALCON_FUNC Box::pack_start_defaults( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    gtk_box_pack_start_defaults( (GtkBox*)_obj, wdt );
}


FALCON_FUNC Box::pack_end_defaults( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    gtk_box_pack_end_defaults( (GtkBox*)_obj, wdt );
}


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


FALCON_FUNC Box::set_homogeneous( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_box_set_homogeneous( (GtkBox*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


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


FALCON_FUNC Box::reorder_child( VMARG )
{
    Item* i_wdt = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget )
        || !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    gtk_box_reorder_child( (GtkBox*)_obj, wdt, i_pos->asInteger() );
}


//FALCON_FUNC Box::query_child_packing( VMARG );

//FALCON_FUNC Box::set_child_packing( VMARG );



} // Gtk
} // Falcon
