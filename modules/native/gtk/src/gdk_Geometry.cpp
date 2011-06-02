/**
 *  \file gdk_Geometry.cpp
 */

#include "gdk_Geometry.hpp"

#undef MYSELF
#define MYSELF Gdk::Geometry* self = Falcon::dyncast<Gdk::Geometry*>( vm->self().asObjectSafe() )

/*#
   @beginmodule gtk
*/


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Geometry::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Geometry = mod->addClass( "GdkGeometry", &Geometry::init );

    //c_Geometry->setWKS( true );
    c_Geometry->getClassDef()->factory( &Geometry::factory );

    mod->addClassProperty( c_Geometry, "min_width" );
    mod->addClassProperty( c_Geometry, "min_height" );
    mod->addClassProperty( c_Geometry, "max_width" );
    mod->addClassProperty( c_Geometry, "max_height" );
    mod->addClassProperty( c_Geometry, "base_width" );
    mod->addClassProperty( c_Geometry, "base_height" );
    mod->addClassProperty( c_Geometry, "width_inc" );
    mod->addClassProperty( c_Geometry, "height_inc" );
    mod->addClassProperty( c_Geometry, "min_aspect" );
    mod->addClassProperty( c_Geometry, "max_aspect" );
    mod->addClassProperty( c_Geometry, "win_gravity" );
}


Geometry::Geometry( const Falcon::CoreClass* gen, const GdkGeometry* geom )
    :
    Gtk::VoidObject( gen )
{
    alloc();
    if ( geom )
        setObject( geom );
}


Geometry::Geometry( const Geometry& other )
    :
    Gtk::VoidObject( other )
{
    alloc();
    if ( other.m_obj )
        setObject( other.m_obj );
}


Geometry::~Geometry()
{
    if ( m_obj )
        memFree( m_obj );
}


void Geometry::alloc()
{
    m_obj = memAlloc( sizeof( GdkGeometry ) );
    memset( m_obj, 0, sizeof( GdkGeometry ) );
}


void Geometry::setObject( const void* geom )
{
    memcpy( m_obj, geom, sizeof( GdkGeometry ) );
}


bool Geometry::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkGeometry* m_geom = (GdkGeometry*) m_obj;

    if ( s == "min_width" )
        it = m_geom->min_width;
    else
    if ( s == "min_height" )
        it = m_geom->min_height;
    else
    if ( s == "max_width" )
        it = m_geom->max_width;
    else
    if ( s == "max_height" )
        it = m_geom->max_height;
    else
    if ( s == "base_width" )
        it = m_geom->base_width;
    else
    if ( s == "base_height" )
        it = m_geom->base_height;
    else
    if ( s == "width_inc" )
        it = m_geom->width_inc;
    else
    if ( s == "height_inc" )
        it = m_geom->height_inc;
    else
    if ( s == "min_aspect" )
        it = m_geom->min_aspect;
    else
    if ( s == "max_aspect" )
        it = m_geom->max_aspect;
    else
    if ( s == "win_gravity" )
        it = (int64) m_geom->win_gravity;
    else
        return defaultProperty( s, it );
    return true;
}


bool Geometry::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    assert( m_obj );
    GdkGeometry* m_geom = (GdkGeometry*) m_obj;

    if ( s == "min_width" )
        m_geom->min_width = it.forceInteger();
    else
    if ( s == "min_height" )
        m_geom->min_height = it.forceInteger();
    else
    if ( s == "max_width" )
        m_geom->max_width = it.forceInteger();
    else
    if ( s == "max_height" )
        m_geom->max_height = it.forceInteger();
    else
    if ( s == "base_width" )
        m_geom->base_width = it.forceInteger();
    else
    if ( s == "base_height" )
        m_geom->base_height = it.forceInteger();
    else
    if ( s == "width_inc" )
        m_geom->width_inc = it.forceInteger();
    else
    if ( s == "height_inc" )
        m_geom->height_inc = it.forceInteger();
    else
    if ( s == "min_aspect" )
        m_geom->min_aspect = it.forceNumeric();
    else
    if ( s == "max_aspect" )
        m_geom->max_aspect = it.forceNumeric();
    else
    if ( s == "win_gravity" )
        m_geom->win_gravity = (GdkGravity) it.forceInteger();
    else
        return false;
    return true;
}


Falcon::CoreObject* Geometry::factory( const Falcon::CoreClass* gen, void* geom, bool )
{
    return new Geometry( gen, (GdkGeometry*) geom );
}


/*#
    @class GdkGeometry
    @brief The GdkGeometry struct gives the window manager information about a window's geometry constraints.
    @optparam min_width minimum width of window (or -1 to use requisition, with GtkWindow only)
    @optparam min_height minimum height of window (or -1 to use requisition, with GtkWindow only)
    @optparam max_width maximum width of window (or -1 to use requisition, with GtkWindow only)
    @optparam max_height maximum height of window (or -1 to use requisition, with GtkWindow only)
    @optparam base_width allowed window widths are base_width + width_inc * N where N is any integer (-1 allowed with GtkWindow)
    @optparam base_height allowed window widths are base_height + height_inc * N where N is any integer (-1 allowed with GtkWindow)
    @optparam width_inc width resize increment
    @optparam height_inc height resize increment
    @optparam min_aspect minimum width/height ratio
    @optparam max_aspect maximum width/height ratio
    @optparam win_gravity window gravity (GdkGravity), see gtk_window_set_gravity()

    Normally you would set these on the GTK+ level using gtk_window_set_geometry_hints().
    GtkWindow then sets the hints on the GdkWindow it creates.

    gdk_window_set_geometry_hints() expects the hints to be fully valid already
    and simply passes them to the window manager; in contrast,
    gtk_window_set_geometry_hints() performs some interpretation. For example,
    GtkWindow will apply the hints to the geometry widget instead of the
    toplevel window, if you set a geometry widget. Also, the
    min_width/min_height/max_width/max_height fields may be set to -1, and
    GtkWindow will substitute the size request of the window or geometry widget.
    If the minimum size hint is not provided, GtkWindow will use its requisition
    as the minimum size. If the minimum size is provided and a geometry widget
    is set, GtkWindow will take the minimum size as the minimum size of the
    geometry widget rather than the entire window. The base size is treated
    similarly.

    The canonical use-case for gtk_window_set_geometry_hints() is to get a
    terminal widget to resize properly. Here, the terminal text area should be
    the geometry widget; GtkWindow will then automatically set the base size to
    the size of other widgets in the terminal window, such as the menubar and
    scrollbar. Then, the width_inc and height_inc fields should be set to the
    size of one character in the terminal. Finally, the base size should be set
    to the size of one character. The net effect is that the minimum size of
    the terminal will have a 1x1 character terminal area, and only terminal
    sizes on the "character grid" will be allowed.

 */
FALCON_FUNC Geometry::init( VMARG )
{
    Gtk::ArgCheck0 args( vm, "[I,I,I,I,I,I,I,I,N,N,GdkGravity]" );
    GdkGeometry hints;
    hints.min_width = args.getInteger( 0, false );
    hints.min_height = args.getInteger( 1, false );
    hints.max_width = args.getInteger( 2, false );
    hints.max_height = args.getInteger( 3, false );
    hints.base_width = args.getInteger( 4, false );
    hints.base_height = args.getInteger( 5, false );
    hints.width_inc = args.getInteger( 6, false );
    hints.height_inc = args.getInteger( 7, false );
    hints.min_aspect = args.getNumeric( 8, false );
    hints.max_aspect = args.getNumeric( 9, false );
    hints.win_gravity = (GdkGravity) args.getInteger( 10, false );
    MYSELF;
    self->setObject( &hints );
}


} // Gdk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
