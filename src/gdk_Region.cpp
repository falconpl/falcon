/**
 *  \file gdk_Region.cpp
 */

#include "gdk_Region.hpp"

#include "gdk_Point.hpp"
#include "gdk_Rectangle.hpp"

#undef MYSELF
#define MYSELF Gdk::Region* self = Falcon::dyncast<Gdk::Region*>( vm->self().asObjectSafe() )


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Region::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Region = mod->addClass( "GdkRegion", &Region::init );

    c_Region->setWKS( true );
    c_Region->getClassDef()->factory( &Region::factory );

    Gtk::MethodTab methods[] =
    {
    { "polygon",        &Region::polygon },
    { "copy",           &Region::copy },
    { "rectangle",      &Region::rectangle },
#if 0 // not used
    { "destroy",        &Region::destroy },
#endif
    { "get_clipbox",    &Region::get_clipbox },
    { "get_rectangles", &Region::get_rectangles },
    { "empty",          &Region::empty },
    { "equal",          &Region::equal },
#if GTK_MINOR_VERSION >= 18
    { "rect_equal",     &Region::rect_equal },
#endif
    { "point_in",       &Region::point_in },
    { "rect_in",        &Region::rect_in },
    { "offset",         &Region::offset },
    { "shrink",         &Region::shrink },
    { "union_with_rect",&Region::union_with_rect },
    { "union_with_rect",&Region::union_with_rect },
    { "union",          &Region::union_ },
    { "subtract",       &Region::subtract },
    { "xor",            &Region::xor_ },
#if 0 // todo
    { "spans_intersect_foreach",&Region::spans_intersect_foreach },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Region, meth->name, meth->cb );
}


Region::Region( const Falcon::CoreClass* gen,
                const GdkRegion* region, const bool transfer )
    :
    Falcon::CoreObject( gen ),
    m_region( NULL )
{
    if ( region )
        m_region = transfer ? (GdkRegion*) region : gdk_region_copy( region );
    else
        m_region = gdk_region_new();
}


Region::~Region()
{
    if ( m_region )
        gdk_region_destroy( m_region );
}


bool Region::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    return defaultProperty( s, it );
}


bool Region::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* Region::factory( const Falcon::CoreClass* gen, void* region, bool )
{
    return new Region( gen, (GdkRegion*) region ); // does not transfer..
}


/*#
    @class GdkRegion
    @brief A GdkRegion represents a set of pixels on the screen.

 */
FALCON_FUNC Region::init( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    // does nothing
}


/*#
    @method polygon GdkRegion
    @brief Creates a new GdkRegion using the polygon defined by a number of points.
    @param points an array of GdkPoint
    @param fill_rule specifies which pixels are included in the region when the polygon overlaps itself (GdkFillRule).
    @return a new GdkRegion based on the given polygon
 */
FALCON_FUNC Region::polygon( VMARG )
{
    const char* spec = "A,GdkFillRule";
    Item* i_points = vm->param( 0 );
    Item* i_rule = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_points || !i_points->isArray()
        || !i_rule || !i_rule->isInteger() )
        throw_inv_params( spec );
#endif
    CoreArray* a_points = i_points->asArray();
    const uint32 cnt = a_points->length();
    GdkRegion* ret = NULL;
    if ( cnt == 0 ) // probably invalid..
        ret = gdk_region_polygon( NULL, 0, (GdkFillRule) i_rule->asInteger() );
    else
    {
        Item pt;
        GdkPoint* tmp;
        GdkPoint* points = (GdkPoint*) memAlloc( cnt * sizeof( GdkPoint ) );
        assert( points );
        for ( uint32 i = 0; i < cnt; ++i )
        {
            pt = a_points->at( i );
#ifndef NO_PARAMETER_CHECK
            if ( !pt.isObject() || !IS_DERIVED( &pt, GdkPoint ) )
            {
                memFree( points );
                throw_inv_params( spec );
            }
#endif
            tmp = dyncast<Gdk::Point*>( pt.asObjectSafe() )->getPoint();
            memcpy( &points[i], tmp, sizeof( GdkPoint ) );
        }
        ret = gdk_region_polygon( points, cnt, (GdkFillRule) i_rule->asInteger() );
        memFree( points );
    }
    assert( ret );
    vm->retval( new Gdk::Region( vm->findWKI( "GdkRegion" )->asClass(), ret,
                                 true ) );
}


/*#
    @method copy GdkRegion
    @brief Copies region, creating an identical new region.
    @return a new region identical to region
 */
FALCON_FUNC Region::copy( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GdkRegion* ret = gdk_region_copy( self->getRegion() );
    vm->retval( new Gdk::Region( vm->findWKI( "GdkRegion" )->asClass(), ret,
                                 true ) );
}


/*#
    @method rectangle GdkRegion
    @brief Creates a new region containing the area rectangle.
    @param rectangle a GdkRectangle
    @return a new region
 */
FALCON_FUNC Region::rectangle( VMARG )
{
    Item* i_rec = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_rec || !i_rec->isObject() || !IS_DERIVED( i_rec, GdkRectangle ) )
        throw_inv_params( "GdkRectangle" );
#endif
    GdkRectangle* rec = dyncast<Gdk::Rectangle*>( i_rec->asObjectSafe() )->getRectangle();
    GdkRegion* ret = gdk_region_rectangle( rec );
    vm->retval( new Gdk::Region( vm->findWKI( "GdkRegion" )->asClass(), ret,
                                 true ) );
}


#if 0 // not used
FALCON_FUNC Region::destroy( VMARG );
#endif


/*#
    @method get_clipbox GdkRegion
    @brief Obtains the smallest rectangle which includes the entire GdkRegion.
    @return a GdkRectangle
 */
FALCON_FUNC Region::get_clipbox( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GdkRectangle rec;
    gdk_region_get_clipbox( self->getRegion(), &rec );
    vm->retval( new Gdk::Rectangle( vm->findWKI( "GdkRectangle" )->asClass(), &rec ) );
}


/*#
    @method get_rectangles GdkRegion
    @brief Obtains the area covered by the region as a list of rectangles.
    @return an array of GdkRectangle
 */
FALCON_FUNC Region::get_rectangles( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GdkRectangle* rects = NULL;
    gint cnt;
    gdk_region_get_rectangles( self->getRegion(), &rects, &cnt );
    CoreArray* arr = new CoreArray( cnt );
    for ( int i = 0; i < cnt; ++i )
    {
        GdkRectangle* rec = &rects[i];
        arr->append( new Gdk::Rectangle( vm->findWKI( "GdkRectangle" )->asClass(),
                                         rec ) );
    }
    if ( rects )
        g_free( rects );
    vm->retval( arr );
}


/*#
    @method empty GdkRegion
    @brief Finds out if the GdkRegion is empty.
    @return TRUE if region is empty.
 */
FALCON_FUNC Region::empty( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    vm->retval( (bool) gdk_region_empty( self->getRegion() ) );
}


/*#
    @method equal GdkRegion
    @brief Finds out if the two regions are the same.
    @param region a GdkRegion
    @return TRUE if this and region are equal.
 */
FALCON_FUNC Region::equal( VMARG )
{
    Item* i_reg = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_reg || !i_reg->isObject() || !IS_DERIVED( i_reg, GdkRegion ) )
        throw_inv_params( "GdkRegion" );
#endif
    GdkRegion* reg = dyncast<Gdk::Region*>( i_reg->asObjectSafe() )->getRegion();
    MYSELF;
    vm->retval( (bool) gdk_region_equal( self->getRegion(), reg ) );
}


#if GTK_MINOR_VERSION >= 18
/*#
    @method rect_equal GdkRegion
    @brief Finds out if a regions is the same as a rectangle.
    @param rectangle a GdkRectangle
    @return TRUE if region and rectangle are equal.
 */
FALCON_FUNC Region::rect_equal( VMARG )
{
    Item* i_rec = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_rec || !i_rec->isObject() || !IS_DERIVED( i_rec, GdkRectangle ) )
        throw_inv_params( "GdkRectangle" );
#endif
    GdkRectangle* rec = dyncast<Gdk::Rectangle*>( i_rec->asObjectSafe() )->getRectangle();
    MYSELF;
    vm->retval( (bool) gdk_region_rect_equal( self->getRegion(), rec ) );
}
#endif


/*#
    @method point_in GdkRegion
    @brief Finds out if a point is in a region.
    @param x the x coordinate of a point
    @param y the y coordinate of a point
    @return TRUE if the point is in region.
 */
FALCON_FUNC Region::point_in( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    vm->retval( (bool) gdk_region_point_in( self->getRegion(),
                                            i_x->asInteger(), i_y->asInteger() ) );
}


/*#
    @method rect_in GdkRegion
    @brief Tests whether a rectangle is within a region.
    @param rectangle a GdkRectangle
    @return GDK_OVERLAP_RECTANGLE_IN, GDK_OVERLAP_RECTANGLE_OUT, or GDK_OVERLAP_RECTANGLE_PART, depending on whether the rectangle is inside, outside, or partly inside the GdkRegion, respectively.
 */
FALCON_FUNC Region::rect_in( VMARG )
{
    Item* i_rec = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_rec || !i_rec->isObject() || !IS_DERIVED( i_rec, GdkRectangle ) )
        throw_inv_params( "GdkRectangle" );
#endif
    GdkRectangle* rec = dyncast<Gdk::Rectangle*>( i_rec->asObjectSafe() )->getRectangle();
    MYSELF;
    vm->retval( (int64) gdk_region_rect_in( self->getRegion(), rec ) );
}


/*#
    @method offset GdkRegion
    @brief Moves a region the specified distance.
    @param dx the distance to move the region horizontally
    @param dy the distance to move the region vertically
 */
FALCON_FUNC Region::offset( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    gdk_region_offset( self->getRegion(), i_x->asInteger(), i_y->asInteger() );
}


/*#
    @method shrink GdkRegion
    @brief Resizes a region by the specified amount. Positive values shrink the region. Negative values expand it.
    @param dx the number of pixels to shrink the region horizontally
    @param dy the number of pixels to shrink the region vertically
 */
FALCON_FUNC Region::shrink( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    gdk_region_shrink( self->getRegion(), i_x->asInteger(), i_y->asInteger() );
}


/*#
    @method union_with_rect GdkRegion
    @brief Sets the area of region to the union of the areas of region and rect.
    @param rect a GdkRectangle

    The resulting area is the set of pixels contained in either region or rect.
 */
FALCON_FUNC Region::union_with_rect( VMARG )
{
    Item* i_rec = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_rec || !i_rec->isObject() || !IS_DERIVED( i_rec, GdkRectangle ) )
        throw_inv_params( "GdkRectangle" );
#endif
    GdkRectangle* rec = dyncast<Gdk::Rectangle*>( i_rec->asObjectSafe() )->getRectangle();
    MYSELF;
    gdk_region_union_with_rect( self->getRegion(), rec );
}


/*#
    @method intersect GdkRegion
    @brief Sets the area of this instance to its intersection with the area of source.
    @param source a GdkRegion

    The resulting area is the set of pixels contained in both this region and source2.
 */
FALCON_FUNC Region::intersect( VMARG )
{
    Item* i_src = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_src || !i_src->isObject() || !IS_DERIVED( i_src, GdkRegion ) )
        throw_inv_params( "GdkRegion" );
#endif
    GdkRegion* src = dyncast<Gdk::Region*>( i_src->asObjectSafe() )->getRegion();
    MYSELF;
    gdk_region_intersect( self->getRegion(), src );
}


/*#
    @method union GdkRegion
    @brief Sets the area of this instance to the union of the areas this region and source.
    @param source a GdkRegion

    The resulting area is the set of pixels contained in either this region or source.
 */
FALCON_FUNC Region::union_( VMARG )
{
    Item* i_src = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_src || !i_src->isObject() || !IS_DERIVED( i_src, GdkRegion ) )
        throw_inv_params( "GdkRegion" );
#endif
    GdkRegion* src = dyncast<Gdk::Region*>( i_src->asObjectSafe() )->getRegion();
    MYSELF;
    gdk_region_union( self->getRegion(), src );
}


/*#
    @method subtract GdkRegion
    @brief Subtracts the area of source from the area of this instance.
    @param source another GdkRegion

    The resulting area is the set of pixels contained in this region but not in source.
 */
FALCON_FUNC Region::subtract( VMARG )
{
    Item* i_src = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_src || !i_src->isObject() || !IS_DERIVED( i_src, GdkRegion ) )
        throw_inv_params( "GdkRegion" );
#endif
    GdkRegion* src = dyncast<Gdk::Region*>( i_src->asObjectSafe() )->getRegion();
    MYSELF;
    gdk_region_subtract( self->getRegion(), src );
}


/*#
    @method xor GdkRegion
    @brief Sets the area of this instance to the exclusive-OR of the areas of this region and source.
    @param source another GdkRegion

    The resulting area is the set of pixels contained in one or the other of
    the two sources but not in both.
 */
FALCON_FUNC Region::xor_( VMARG )
{
    Item* i_src = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_src || !i_src->isObject() || !IS_DERIVED( i_src, GdkRegion ) )
        throw_inv_params( "GdkRegion" );
#endif
    GdkRegion* src = dyncast<Gdk::Region*>( i_src->asObjectSafe() )->getRegion();
    MYSELF;
    gdk_region_xor( self->getRegion(), src );
}


//FALCON_FUNC Region::spans_intersect_foreach( VMARG );


} // Gdk
} // Falcon
