/**
 *  \file gdk_Color.cpp
 */

#include "gdk_Color.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Color::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Color = mod->addClass( "GdkColor" );

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
    if ( clr )
        m_obj = gdk_color_copy( clr );
}


Color::Color( const Color& other )
    :
    Gtk::VoidObject( other )
{
    m_obj = gdk_color_copy( (GdkColor*) other.m_obj );
}


Color::~Color()
{
    if ( m_obj )
        gdk_color_free( (GdkColor*) m_obj );
}


void Color::setObject( const void* clr )
{
    VoidObject::setObject( clr );
    m_obj = gdk_color_copy( (GdkColor*) clr );
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

    @prop pixel For allocated colors, the value used to draw this color on the screen.
    @prop red The red component of the color. This is a value between 0 and 65535, with 65535 indicating full intensity.
    @prop green The green component of the color.
    @prop blue The blue component of the color.
 */

} // Gdk
} // Falcon
