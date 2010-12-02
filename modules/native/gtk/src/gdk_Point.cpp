/**
 *  \file gdk_Point.cpp
 */

#include "gdk_Point.hpp"


#undef MYSELF
#define MYSELF Gdk::Point* self = Falcon::dyncast<Gdk::Point*>( vm->self().asObjectSafe() )

/*#
   @beginmodule gtk
*/


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Point::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Point = mod->addClass( "GdkPoint", &Point::init );

    c_Point->setWKS( true );
    c_Point->getClassDef()->factory( &Point::factory );

    mod->addClassProperty( c_Point, "x" );
    mod->addClassProperty( c_Point, "y" );
}


Point::Point( const Falcon::CoreClass* gen, const GdkPoint* point )
    :
    Gtk::VoidObject( gen )
{
    if ( point )
        setObject( point );
}


Point::Point( const Point& other )
    :
    Gtk::VoidObject( other )
{
    m_obj = 0;
    if ( other.m_obj )
        setObject( other.m_obj );
}


Point::~Point()
{
    if ( m_obj )
        memFree( m_obj );
}


void Point::alloc()
{
    assert( m_obj == 0 );
    m_obj = memAlloc( sizeof( GdkPoint ) );
}

void Point::setObject( const void* pt )
{
    assert( m_obj == 0 );
    alloc();
    memcpy( m_obj, pt, sizeof( GdkPoint ) );
}


bool Point::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkPoint* m_point = (GdkPoint*) m_obj;

    if ( s == "x" )
        it = m_point->x;
    else
    if ( s == "y" )
        it = m_point->y;
    else
        return defaultProperty( s, it );
    return true;
}


bool Point::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    assert( m_obj );
    GdkPoint* m_point = (GdkPoint*) m_obj;

    if ( s == "x" )
        m_point->x = it.forceInteger();
    else
    if ( s == "y" )
        m_point->y = it.forceInteger();
    else
        return false;
    return true;
}


Falcon::CoreObject* Point::factory( const Falcon::CoreClass* gen, void* point, bool )
{
    return new Point( gen, (GdkPoint*) point );
}


/*#
    @class GdkPoint
    @brief Defines the x and y coordinates of a point.
    @optparam x the x coordinate of the point.
    @optparam y the y coordinate of the point.

    @prop x the x coordinate of the point.
    @prop y the y coordinate of the point.

 */
FALCON_FUNC Point::init( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( ( i_x && !i_x->isInteger() )
        || ( i_y && !i_y->isInteger() ) )
        throw_inv_params( "[I,I]" );
#endif
    MYSELF;
    self->alloc();
    ((GdkPoint*)self->m_obj)->x = i_x ? i_x->asInteger() : 0;
    ((GdkPoint*)self->m_obj)->y = i_y ? i_y->asInteger() : 0;
}


} // Gdk
} // Falcon
