/**
 *  \file gdk_GCValues.cpp
 */

#include "gdk_GCValues.hpp"

#include "gdk_Color.hpp"
#include "gdk_Pixmap.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void GCValues::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_GCValues = mod->addClass( "GdkGCValues" );

    c_GCValues->setWKS( true );
    c_GCValues->getClassDef()->factory( &GCValues::factory );

    mod->addClassProperty( c_GCValues, "foreground" );
    mod->addClassProperty( c_GCValues, "background" );
    mod->addClassProperty( c_GCValues, "font" );
    mod->addClassProperty( c_GCValues, "function" );
    mod->addClassProperty( c_GCValues, "fill" );
    mod->addClassProperty( c_GCValues, "tile" );
    mod->addClassProperty( c_GCValues, "stipple" );
    mod->addClassProperty( c_GCValues, "clip_mask" );
    mod->addClassProperty( c_GCValues, "subwindow_mode" );
    mod->addClassProperty( c_GCValues, "ts_x_origin" );
    mod->addClassProperty( c_GCValues, "ts_y_origin" );
    mod->addClassProperty( c_GCValues, "clip_x_origin" );
    mod->addClassProperty( c_GCValues, "clip_y_origin" );
    mod->addClassProperty( c_GCValues, "graphics_exposures" );
    mod->addClassProperty( c_GCValues, "line_width" );
    mod->addClassProperty( c_GCValues, "line_style" );
    mod->addClassProperty( c_GCValues, "cap_style" );
    mod->addClassProperty( c_GCValues, "join_style" );
}


GCValues::GCValues( const Falcon::CoreClass* gen, const GdkGCValues* gcvalues )
    :
    Gtk::VoidObject( gen )
{
    if ( gcvalues )
        setObject( gcvalues );
}


GCValues::GCValues( const GCValues& other )
    :
    Gtk::VoidObject( other )
{
    m_obj = 0;
    if ( other.m_obj )
        setObject( other.m_obj );
}


GCValues::~GCValues()
{
    if ( m_obj )
    {
        decref();
        memFree( m_obj );
    }
}


void GCValues::incref()
{
    assert( m_obj );
    GdkGCValues* m_gcvalues = (GdkGCValues*) m_obj;

    if ( m_gcvalues->font )
        gdk_font_ref( m_gcvalues->font );
    if ( m_gcvalues->tile )
        g_object_ref_sink( (GObject*) m_gcvalues->tile );
    if ( m_gcvalues->stipple )
        g_object_ref_sink( (GObject*) m_gcvalues->stipple );
    if ( m_gcvalues->clip_mask )
        g_object_ref_sink( (GObject*) m_gcvalues->clip_mask );
}


void GCValues::decref()
{
    assert( m_obj );
    GdkGCValues* m_gcvalues = (GdkGCValues*) m_obj;

    if ( m_gcvalues->font )
        gdk_font_unref( m_gcvalues->font );
    if ( m_gcvalues->tile )
        g_object_unref( (GObject*) m_gcvalues->tile );
    if ( m_gcvalues->stipple )
        g_object_unref( (GObject*) m_gcvalues->stipple );
    if ( m_gcvalues->clip_mask )
        g_object_unref( (GObject*) m_gcvalues->clip_mask );
}


void GCValues::setObject( const void* gcvalues )
{
    assert( m_obj == 0 );
    m_obj = memAlloc( sizeof( GdkGCValues ) );
    memcpy( m_obj, gcvalues, sizeof( GdkGCValues ) );
    incref();
}


bool GCValues::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkGCValues* m_gcvalues = (GdkGCValues*) m_obj;
    VMachine* vm = VMachine::getCurrent();

    if ( s == "foreground" )
        it = new Gdk::Color( vm->findWKI( "GdkColor" )->asClass(),
                             &m_gcvalues->foreground );
    else
    if ( s == "background" )
        it = new Gdk::Color( vm->findWKI( "GdkColor" )->asClass(),
                             &m_gcvalues->background );
#if 0 // todo
    else
    if ( s == "font" )
        it = ;
#endif
    else
    if ( s == "function" )
        it = (int64) m_gcvalues->function;
    else
    if ( s == "fill" )
        it = (int64) m_gcvalues->fill;
    else
    if ( s == "tile" )
        it = new Gdk::Pixmap( vm->findWKI( "GdkPixmap" )->asClass(),
                              m_gcvalues->tile );
    else
    if ( s == "stipple" )
        it = new Gdk::Pixmap( vm->findWKI( "GdkPixmap" )->asClass(),
                              m_gcvalues->stipple );
    else
    if ( s == "clip_mask" )
        it = new Gdk::Pixmap( vm->findWKI( "GdkPixmap" )->asClass(),
                              m_gcvalues->clip_mask );
    else
    if ( s == "subwindow_mode" )
        it = (int64) m_gcvalues->subwindow_mode;
    else
    if ( s == "ts_x_origin" )
        it = m_gcvalues->ts_y_origin;
    else
    if ( s == "ts_y_origin" )
        it = m_gcvalues->ts_y_origin;
    else
    if ( s == "clip_x_origin" )
        it = m_gcvalues->clip_x_origin;
    else
    if ( s == "clip_y_origin" )
        it = m_gcvalues->clip_y_origin;
    else
    if ( s == "graphics_exposures" )
        it = m_gcvalues->graphics_exposures;
    else
    if ( s == "line_width" )
        it = m_gcvalues->line_width;
    else
    if ( s == "line_style" )
        it = (int64) m_gcvalues->line_style;
    else
    if ( s == "cap_style" )
        it = (int64) m_gcvalues->cap_style;
    else
    if ( s == "join_style" )
        it = (int64) m_gcvalues->join_style;
    else
        return false;
    return true;
}


bool GCValues::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    assert( m_obj );
    GdkGCValues* m_gcvalues = (GdkGCValues*) m_obj;

    if ( s == "foreground" )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !it.isObject() || !IS_DERIVED( &it, GdkColor ) )
            throw_inv_params( "GdkColor" );
#endif
        GdkColor* clr = GET_COLOR( it );
        memcpy( &m_gcvalues->foreground, clr, sizeof( GdkColor ) );
    }
    else
    if ( s == "background" )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !it.isObject() || !IS_DERIVED( &it, GdkColor ) )
            throw_inv_params( "GdkColor" );
#endif
        GdkColor* clr = GET_COLOR( it );
        memcpy( &m_gcvalues->background, clr, sizeof( GdkColor ) );
    }
#if 0 // todo
    else
    if ( s == "font" )
        ;
#endif
    else
    if ( s == "function" )
        m_gcvalues->function = (GdkFunction) it.forceInteger();
    else
    if ( s == "fill" )
        m_gcvalues->fill = (GdkFill) it.forceInteger();
    else
    if ( s == "tile" )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !it.isObject() || !IS_DERIVED( &it, GdkPixmap ) )
            throw_inv_params( "GdkPixmap" );
#endif
        GdkPixmap* pix = (GdkPixmap*) COREGOBJECT( &it )->getObject();
        if ( m_gcvalues->tile )
            g_object_unref( (GObject*) m_gcvalues->tile );
        m_gcvalues->tile = pix;
        g_object_ref( (GObject*) m_gcvalues->tile );
    }
    else
    if ( s == "stipple" )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !it.isObject() || !IS_DERIVED( &it, GdkPixmap ) )
            throw_inv_params( "GdkPixmap" );
#endif
        GdkPixmap* pix = (GdkPixmap*) COREGOBJECT( &it )->getObject();
        if ( m_gcvalues->stipple )
            g_object_unref( (GObject*) m_gcvalues->stipple );
        m_gcvalues->stipple = pix;
        g_object_ref( (GObject*) m_gcvalues->stipple );
    }
    else
    if ( s == "clip_mask" )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !it.isObject() || !IS_DERIVED( &it, GdkPixmap ) )
            throw_inv_params( "GdkPixmap" );
#endif
        GdkPixmap* pix = (GdkPixmap*) COREGOBJECT( &it )->getObject();
        if ( m_gcvalues->clip_mask )
            g_object_unref( (GObject*) m_gcvalues->clip_mask );
        m_gcvalues->clip_mask = pix;
        g_object_ref( (GObject*) m_gcvalues->clip_mask );
    }
    else
    if ( s == "subwindow_mode" )
        m_gcvalues->subwindow_mode = (GdkSubwindowMode) it.forceInteger();
    else
    if ( s == "ts_x_origin" )
        m_gcvalues->ts_y_origin = it.forceInteger();
    else
    if ( s == "ts_y_origin" )
        m_gcvalues->ts_y_origin = it.forceInteger();
    else
    if ( s == "clip_x_origin" )
        m_gcvalues->clip_x_origin = it.forceInteger();
    else
    if ( s == "clip_y_origin" )
        m_gcvalues->clip_y_origin = it.forceInteger();
    else
    if ( s == "graphics_exposures" )
        m_gcvalues->graphics_exposures = it.forceInteger();
    else
    if ( s == "line_width" )
        m_gcvalues->line_width = it.forceInteger();
    else
    if ( s == "line_style" )
        m_gcvalues->line_style = (GdkLineStyle) it.forceInteger();
    else
    if ( s == "cap_style" )
        m_gcvalues->cap_style = (GdkCapStyle) it.forceInteger();
    else
    if ( s == "join_style" )
        m_gcvalues->join_style = (GdkJoinStyle) it.forceInteger();
    else
        return false;
    return true;
}


Falcon::CoreObject* GCValues::factory( const Falcon::CoreClass* gen, void* gcvalues, bool )
{
    return new GCValues( gen, (GdkGCValues*) gcvalues );
}


/*#
    @class GdkGCValues
    @brief The GdkGCValues structure holds a set of values used to create or modify a graphics context.

    @prop foreground the foreground color (GdkColor). Note that gdk_gc_get_values() only sets the pixel value.
    @prop background the background color (GdkColor). Note that gdk_gc_get_values() only sets the pixel value.
    @prop font the default font (GdkFont). TODO
    @prop function the bitwise operation used when drawing (GdkFunction).
    @prop fill the fill style (GdkFill).
    @prop tile the tile pixmap (GdkPixmap).
    @prop stipple the stipple bitmap (GdkPixmap).
    @prop clip_mask the clip mask bitmap (GdkPixmap).
    @prop subwindow_mode the subwindow mode (GdkSubwindowMode).
    @prop ts_x_origin the x origin of the tile or stipple.
    @prop ts_y_origin the y origin of the tile or stipple.
    @prop clip_x_origin the x origin of the clip mask.
    @prop clip_y_origin the y origin of the clip mask.
    @prop graphics_exposures whether graphics exposures are enabled.
    @prop line_width the line width.
    @prop line_style the way dashed lines are drawn (GdkLineStyle).
    @prop cap_style the way the ends of lines are drawn (GdkJoinStyle).
    @prop join_style the way joins between lines are drawn (GdkJoinStyle).
 */


} // Gdk
} // Falcon
