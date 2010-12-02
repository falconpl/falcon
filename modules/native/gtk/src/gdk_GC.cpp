/**
 *  \file gdk_GC.cpp
 */

#include "gdk_GC.hpp"

#include "gdk_GCValues.hpp"
#include "gdk_Screen.hpp"

/*#
   @beginmodule gtk
*/


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void GC::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_GC = mod->addClass( "GdkGC", &GC::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_GC->getClassDef()->addInheritance( in );

    c_GC->setWKS( true );
    c_GC->getClassDef()->factory( &GC::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_with_values",        &GC::new_with_values },
    { "get_screen",             &GC::get_screen },
#if 0 // unused
    { "ref",                    &GC::ref },
    { "unref",                  &GC::unref },
    { "destroy",                &GC::destroy },
#endif
    { "set_values",             &GC::set_values },
    { "get_values",             &GC::get_values },
#if 0
    { "set_foreground",         &GC::set_foreground },
    { "set_background",         &GC::set_background },
    { "set_rgb_fg_color",       &GC::set_rgb_fg_color },
    { "set_rgb_bg_color",       &GC::set_rgb_bg_color },
    { "set_font",               &GC::set_font },
    { "set_function",           &GC::set_function },
    { "set_fill",               &GC::set_fill },
    { "set_tile",               &GC::set_tile },
    { "set_stipple",            &GC::set_stipple },
    { "set_ts_origin",          &GC::set_ts_origin },
    { "set_clip_origin",        &GC::set_clip_origin },
    { "set_clip_mask",          &GC::set_clip_mask },
    { "set_clip_rectangle",     &GC::set_clip_rectangle },
    { "set_clip_region",        &GC::set_clip_region },
    { "set_subwindow",          &GC::set_subwindow },
    { "set_exposures",          &GC::set_exposures },
    { "set_line_attributes",    &GC::set_line_attributes },
    { "set_dashes",             &GC::set_dashes },
    { "copy",                   &GC::copy },
    { "set_colormap",           &GC::set_colormap },
    { "get_colormap",           &GC::get_colormap },
    { "offset",                 &GC::offset },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_GC, meth->name, meth->cb );
}


GC::GC( const Falcon::CoreClass* gen, const GdkGC* gc )
    :
    Gtk::CoreGObject( gen, (GObject*) gc )
{}


Falcon::CoreObject* GC::factory( const Falcon::CoreClass* gen, void* gc, bool )
{
    return new GC( gen, (GdkGC*) gc );
}


/*#
    @class GdkGC
    @brief Objects to encapsulate drawing properties
    @param drawable a GdkDrawable. The created GC must always be used with drawables of the same depth as this one.

    All drawing operations in GDK take a graphics context (GC) argument. A graph
    ics context encapsulates information about the way things are drawn, such as
    the foreground color or line width. By using graphics contexts, the number
    of arguments to each drawing call is greatly reduced, and communication
    overhead is minimized, since identical arguments do not need to be passed
    repeatedly.

    Most values of a graphics context can be set at creation time by using
    gdk_gc_new_with_values(), or can be set one-by-one using functions such as
    gdk_gc_set_foreground(). A few of the values in the GC, such as the dash
    pattern, can only be set by the latter method.
 */
FALCON_FUNC GC::init( VMARG )
{
    Item* i_draw = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_draw || !i_draw->isObject() || !IS_DERIVED( i_draw, GdkDrawable ) )
        throw_inv_params( "GdkDrawable" );
#endif
    GdkDrawable* draw = (GdkDrawable*) COREGOBJECT( i_draw )->getObject();
    MYSELF;
    self->setObject( (GObject*) gdk_gc_new( draw ) );
}


/*#
    @method new_with_values GdkGC
    @brief Create a new GC with the given initial values.
    @param drawable a GdkDrawable. The created GC must always be used with drawables of the same depth as this one.
    @param values a structure containing initial values for the GC (GdkGCValues).
    @param values_mask a bit mask indicating which fields in values are set (GdkGCValuesMask).
    @return the new graphics context.
 */
FALCON_FUNC GC::new_with_values( VMARG )
{
    Item* i_draw = vm->param( 0 );
    Item* i_val = vm->param( 1 );
    Item* i_mask = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_draw || !i_draw->isObject() || !IS_DERIVED( i_draw, GdkDrawable )
        || !i_val || !i_val->isObject() || !IS_DERIVED( i_val, GdkGCValues )
        || !i_mask || !i_mask->isInteger() )
        throw_inv_params( "GdkGCValues,GdkGCValuesMask" );
#endif
    GdkDrawable* draw = (GdkDrawable*) COREGOBJECT( i_draw )->getObject();
    GdkGCValues* val = GET_GCVALUES( *i_val );
    GdkGC* gc = gdk_gc_new_with_values( draw, val, (GdkGCValuesMask) i_mask->asInteger() );
    vm->retval( new Gdk::GC( vm->findWKI( "GdkGC" )->asClass(), gc ) );
}


/*#
    @method get_screen GdkGC
    @brief Gets the GdkScreen for which gc was created
    @return the GdkScreen for gc.
 */
FALCON_FUNC GC::get_screen( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkScreen* screen = gdk_gc_get_screen( (GdkGC*)_obj );
    vm->retval( new Gdk::Screen( vm->findWKI( "GdkScreen" )->asClass(), screen ) );
}


#if 0 // not used
FALCON_FUNC GC::ref( VMARG );
FALCON_FUNC GC::unref( VMARG );
FALCON_FUNC GC::destroy( VMARG );
#endif


/*#
    @method set_values GdkGC
    @brief Sets attributes of a graphics context in bulk.
    @param values struct containing the new values (GdkGCValues).
    @param values_mask mask indicating which struct fields are to be used (GdkGCValuesMask).

    For each flag set in values_mask, the corresponding field will be read from
    values and set as the new value for gc. If you're only setting a few values
    on gc, calling individual "setter" functions is likely more convenient.
 */
FALCON_FUNC GC::set_values( VMARG )
{
    Item* i_val = vm->param( 0 );
    Item* i_mask = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_val || !i_val->isObject() || !IS_DERIVED( i_val, GdkGCValues )
        || !i_mask || !i_mask->isInteger() )
        throw_inv_params( "GdkGCValues,GdkGCValuesMask" );
#endif
    GdkGCValues* val = GET_GCVALUES( *i_val );
    MYSELF;
    GET_OBJ( self );
    gdk_gc_set_values( (GdkGC*)_obj, val, (GdkGCValuesMask) i_mask->asInteger() );
}


/*#
    @method get_values GdkGC
    @brief Retrieves the current values from a graphics context.
    @return the GdkGCValues structure in which to store the results.

    Note that only the pixel values of the values->foreground and
    values->background are filled, use gdk_colormap_query_color() to obtain the
    rgb values if you need them.
 */
FALCON_FUNC GC::get_values( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkGCValues val;
    gdk_gc_get_values( (GdkGC*)_obj, &val );
    vm->retval( new Gdk::GCValues( vm->findWKI( "GdkGCValues" )->asClass(), &val ) );
}


#if 0
FALCON_FUNC GC::set_foreground( VMARG );
FALCON_FUNC GC::set_background( VMARG );
FALCON_FUNC GC::set_rgb_fg_color( VMARG );
FALCON_FUNC GC::set_rgb_bg_color( VMARG );
FALCON_FUNC GC::set_font( VMARG );
FALCON_FUNC GC::set_function( VMARG );
FALCON_FUNC GC::set_fill( VMARG );
FALCON_FUNC GC::set_tile( VMARG );
FALCON_FUNC GC::set_stipple( VMARG );
FALCON_FUNC GC::set_ts_origin( VMARG );
FALCON_FUNC GC::set_clip_origin( VMARG );
FALCON_FUNC GC::set_clip_mask( VMARG );
FALCON_FUNC GC::set_clip_rectangle( VMARG );
FALCON_FUNC GC::set_clip_region( VMARG );
FALCON_FUNC GC::set_subwindow( VMARG );
FALCON_FUNC GC::set_exposures( VMARG );
FALCON_FUNC GC::set_line_attributes( VMARG );
FALCON_FUNC GC::set_dashes( VMARG );
FALCON_FUNC GC::copy( VMARG );
FALCON_FUNC GC::set_colormap( VMARG );
FALCON_FUNC GC::get_colormap( VMARG );
FALCON_FUNC GC::offset( VMARG );
#endif

} // Gdk
} // Falcon
