/**
 *  \file gdk_GC.cpp
 */

#include "gdk_GC.hpp"


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

    //c_GC->setWKS( true );
    c_GC->getClassDef()->factory( &GC::factory );

    Gtk::MethodTab methods[] =
    {
#if 0
    { "new_with_values",        &GC::new_with_values },
    { "get_screen",             &GC::get_screen },
    { "ref",                    &GC::ref },
    { "unref",                  &GC::unref },
    { "destroy",                &GC::destroy },
    { "set_values",             &GC::set_values },
    { "get_values",             &GC::get_values },
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
    GdkDrawable* draw = (GdkDrawable*) COREGOBJECT( i_draw )->getGObject();
    MYSELF;
    self->setGObject( (GObject*) gdk_gc_new( draw ) );
}


#if 0
FALCON_FUNC GC::new_with_values( VMARG );
FALCON_FUNC GC::get_screen( VMARG );
FALCON_FUNC GC::ref( VMARG );
FALCON_FUNC GC::unref( VMARG );
FALCON_FUNC GC::destroy( VMARG );
FALCON_FUNC GC::set_values( VMARG );
FALCON_FUNC GC::get_values( VMARG );
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
