#ifndef GDK_COLOR_HPP
#define GDK_COLOR_HPP

#include "modgtk.hpp"

#define GET_COLOR( item ) \
        (((Gdk::Color*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Color
 */
class Color
    :
    public Gtk::VoidObject
{
public:

    Color( const Falcon::CoreClass*, const GdkColor* = 0 );

    Color( const Color& );

    ~Color();

    Color* clone() const { return new Color( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkColor* getObject() const { return (GdkColor*) m_obj; }

    void setObject( const void* );

    static FALCON_FUNC init( VMARG );

private:

    void alloc();

};


} // Gdk
} // Falcon

#endif // !GDK_COLOR_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
