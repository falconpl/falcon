/**
 *  \file gdk_Rectangle.cpp
 */

#include "gdk_Rectangle.hpp"


#undef MYSELF
#define MYSELF Gdk::Rectangle* self = Falcon::dyncast<Gdk::Rectangle*>( vm->self().asObjectSafe() )


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Rectangle::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Rectangle = mod->addClass( "GdkRectangle", &Rectangle::init );

    c_Rectangle->setWKS( true );
    c_Rectangle->getClassDef()->factory( &Rectangle::factory );

    mod->addClassProperty( c_Rectangle, "x" );
    mod->addClassProperty( c_Rectangle, "y" );
    mod->addClassProperty( c_Rectangle, "width" );
    mod->addClassProperty( c_Rectangle, "height" );

    mod->addClassMethod( c_Rectangle, "intersect",  &Rectangle::intersect );
    mod->addClassMethod( c_Rectangle, "union",      &Rectangle::union_ );
}


Rectangle::Rectangle( const Falcon::CoreClass* gen, const GdkRectangle* rect )
    :
    Falcon::CoreObject( gen )
{
    if ( rect )
        memcpy( &m_rectangle, rect, sizeof( GdkRectangle ) );
    else
        memset( &m_rectangle, 0, sizeof( GdkRectangle ) );
}


Rectangle::~Rectangle()
{
}


bool Rectangle::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    if ( s == "x" )
        it = m_rectangle.x;
    else
    if ( s == "y" )
        it = m_rectangle.y;
    else
    if ( s == "width" )
        it = m_rectangle.width;
    else
    if ( s == "height" )
        it = m_rectangle.height;
    else
        return defaultProperty( s, it );
    return true;
}


bool Rectangle::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    if ( s == "x" )
        m_rectangle.x = it.forceInteger();
    else
    if ( s == "y" )
        m_rectangle.y = it.forceInteger();
    else
    if ( s == "width" )
        m_rectangle.width = it.forceInteger();
    else
    if ( s == "height" )
        m_rectangle.height = it.forceInteger();
    else
        return false;
    return true;
}


Falcon::CoreObject* Rectangle::factory( const Falcon::CoreClass* gen, void* rect, bool )
{
    return new Rectangle( gen, (GdkRectangle*) rect );
}


/*#
    @class GdkRectangle
    @brief GdkRectangle is a structure holding the position and size of a rectangle.

    @optparam x the x coordinate of the left edge of the rectangle.
    @optparam y the y coordinate of the top of the rectangle.
    @optparam width the width of the rectangle.
    @optparam height the height of the rectangle.

    @prop x the x coordinate of the left edge of the rectangle.
    @prop y the y coordinate of the top of the rectangle.
    @prop width the width of the rectangle.
    @prop height the height of the rectangle.

    The intersection of two rectangles can be computed with gdk_rectangle_intersect().
    To find the union of two rectangles use gdk_rectangle_union().
 */
FALCON_FUNC Rectangle::init( VMARG )
{
    Gtk::ArgCheck0 args( vm, "[I,I,I,I]" );
    MYSELF;
    self->m_rectangle.x = args.getInteger( 0, false );
    self->m_rectangle.y = args.getInteger( 1, false );
    self->m_rectangle.width = args.getInteger( 2, false );
    self->m_rectangle.height = args.getInteger( 3, false );
}


/*#
    @method intersect GdkRectangle
    @brief Calculates the intersection of two rectangles.
    @param src a GdkRectangle
    @return a GdkRectangle that is the intersection of this rectangle and the GdkRectangle specified by src.

    If the rectangles do not intersect the returned GdkRectangle will have
    all attributes set to 0.
 */
FALCON_FUNC Rectangle::intersect( VMARG )
{
    Item* i_src = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_src || !i_src->isObject() || !IS_DERIVED( i_src, GdkRectangle ) )
        throw_inv_params( "GdkRectangle" );
#endif
    MYSELF;
    GdkRectangle* src1 = self->getRectangle();
    GdkRectangle* src2 = dyncast<Gdk::Rectangle*>( i_src->asObjectSafe() )->getRectangle();
    GdkRectangle dest;
    gboolean ret = gdk_rectangle_intersect( src1, src2, &dest );
    if ( !ret )
        memset( &dest, 0, sizeof( GdkRectangle ) );
    vm->retval( new Gdk::Rectangle( vm->findWKI( "GdkRectangle" )->asClass(), &dest ) );
}


/*#
    @method union GdkRectangle
    @brief Calculates the union of two rectangles.
    @param src a GdkRectangle.
    @return a GdkRectangle that is the smallest rectangle containing both this rectangle and the GdkRectangle specified by src.
 */
FALCON_FUNC Rectangle::union_( VMARG )
{
    Item* i_src = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_src || !i_src->isObject() || !IS_DERIVED( i_src, GdkRectangle ) )
        throw_inv_params( "GdkRectangle" );
#endif
    MYSELF;
    GdkRectangle* src1 = self->getRectangle();
    GdkRectangle* src2 = dyncast<Gdk::Rectangle*>( i_src->asObjectSafe() )->getRectangle();
    GdkRectangle dest;
    gdk_rectangle_union( src1, src2, &dest );
    vm->retval( new Gdk::Rectangle( vm->findWKI( "GdkRectangle" )->asClass(), &dest ) );
}


} // Gdk
} // Falcon
