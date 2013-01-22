/**
 *  \file gdk_Color.cpp
 */

#include "gdk_Color.hpp"

#undef MYSELF
#define MYSELF Gdk::Color* self = Falcon::dyncast<Gdk::Color*>( vm->self().asObjectSafe() )

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Color::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Color = mod->addClass( "GdkColor", &Color::init );

    c_Color->setWKS( true );
    c_Color->getClassDef()->factory( &Color::factory );

    mod->addClassProperty( c_Color, "pixel" );
    mod->addClassProperty( c_Color, "red" );
    mod->addClassProperty( c_Color, "green" );
    mod->addClassProperty( c_Color, "blue" );
}


Color::Color( const Falcon::CoreClass* gen, const GdkColor* clr )
    :
    Gtk::VoidObject( gen )
{
    alloc();
    if ( clr )
        setObject( clr );
}


Color::Color( const Color& other )
    :
    Gtk::VoidObject( other )
{
    m_obj = 0;
    alloc();
    if ( other.m_obj )
        setObject( other.m_obj );
}


Color::~Color()
{
    if ( m_obj )
        free( m_obj );
}


void Color::alloc()
{
    assert( m_obj == 0 );
    m_obj = malloc( sizeof( GdkColor ) );
}


void Color::setObject( const void* clr )
{
    assert( m_obj != 0 );
    memcpy( m_obj, clr, sizeof( GdkColor ) );
}


bool Color::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkColor* m_color = (GdkColor*) m_obj;

    if ( s == "pixel" )
        it = m_color->pixel;
    else
    if ( s == "red" )
        it = m_color->red;
    else
    if ( s == "green" )
        it = m_color->green;
    else
    if ( s == "blue" )
        it = m_color->blue;
    else
        return false;
    return true;
}


bool Color::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    assert( m_obj );
    GdkColor* m_color = (GdkColor*) m_obj;

    if ( s == "pixel" )
        m_color->pixel = it.forceInteger();
    else
    if ( s == "red" )
        m_color->red = it.forceInteger();
    else
    if ( s == "green" )
        m_color->green = it.forceInteger();
    else
    if ( s == "blue" )
        m_color->blue = it.forceInteger();
    else
        return false;
    return true;
}


Falcon::CoreObject* Color::factory( const Falcon::CoreClass* gen, void* clr, bool )
{
    return new Color( gen, (GdkColor*) clr );
}


/*#
    @class GdkColor
    @brief The GdkColor structure is used to describe an allocated or unallocated color.
    @optparam pixel For allocated colors, the value used to draw this color on the screen.
    @optparam red The red component of the color. This is a value between 0 and 65535, with 65535 indicating full intensitiy.
    @optparam green The green component of the color.
    @optparam blue The blue component of the color.

    @prop pixel For allocated colors, the value used to draw this color on the screen.
    @prop red The red component of the color. This is a value between 0 and 65535, with 65535 indicating full intensity.
    @prop green The green component of the color.
    @prop blue The blue component of the color.
 */
FALCON_FUNC Color::init( VMARG )
{
    Item* i_pix = vm->param( 0 );
    Item* i_red = vm->param( 1 );
    Item* i_green = vm->param( 2 );
    Item* i_blue = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( ( i_pix && !i_pix->isInteger() )
        || ( i_red && !i_red->isInteger() )
        || ( i_green && !i_green->isInteger() )
        || ( i_blue && !i_blue->isInteger() ) )
        throw_inv_params( "I,I,I,I" );
#endif
    MYSELF;
    ((GdkColor*)self->m_obj)->pixel = i_pix ? i_pix->asInteger() : 0;
    ((GdkColor*)self->m_obj)->red = i_red ? i_red->asInteger() : 0;
    ((GdkColor*)self->m_obj)->green = i_green ? i_green->asInteger() : 0;
    ((GdkColor*)self->m_obj)->blue = i_blue ? i_blue->asInteger() : 0;
}


} // Gdk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
