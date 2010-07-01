#ifndef GDK_COLOR_HPP
#define GDK_COLOR_HPP

#include "modgtk.hpp"

#define GET_COLOR( item ) \
        (((Gdk::Color*) (item).asObjectSafe() )->getColor())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Color
 */
class Color
    :
    public Falcon::CoreObject
{
public:

    Color( const Falcon::CoreClass*, const GdkColor* = 0 );

    ~Color();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkColor* getColor() const { return m_color; }

private:

    GdkColor*   m_color;

};


} // Gdk
} // Falcon

#endif // !GDK_COLOR_HPP
