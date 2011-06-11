#ifndef GDK_BITMAP_HPP
#define GDK_BITMAP_HPP

#include "modgtk.hpp"

#define GET_BITMAP( item ) \
        (((Gdk::Bitmap*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Bitmap
 */
class Bitmap
    :
    public Gtk::CoreGObject
{
public:

    Bitmap( const Falcon::CoreClass*, const GdkBitmap* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkBitmap* getObject() const { return (GdkBitmap*) m_obj; }

    static FALCON_FUNC create_from_data( VMARG );

};


} // Gdk
} // Falcon

#endif // !GDK_BITMAP_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
