#ifndef GDK_CURSOR_HPP
#define GDK_CURSOR_HPP

#include "modgtk.hpp"

#define GET_CURSOR( item ) \
        (((Gdk::Cursor*) (item).asObjectSafe())->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Cursor
 */
class Cursor
    :
    public Gtk::VoidObject
{
public:

    Cursor( const Falcon::CoreClass*, const GdkCursor* = 0 );

    Cursor( const Cursor& );

    ~Cursor();

    Cursor* clone() const { return new Cursor( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkCursor* getObject() const { return (GdkCursor*) m_obj; }

    void setObject( const void* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_from_pixmap( VMARG );

    static FALCON_FUNC new_from_pixbuf( VMARG );

    static FALCON_FUNC new_from_name( VMARG );

    static FALCON_FUNC new_for_display( VMARG );

    static FALCON_FUNC get_display( VMARG );

    static FALCON_FUNC get_image( VMARG );

#if 0 // not used
    static FALCON_FUNC ref( VMARG );
    static FALCON_FUNC unref( VMARG );
    static FALCON_FUNC destroy( VMARG );
#endif

private:

    void incref() const;

    void decref() const;

};


} // Gdk
} // Falcon

#endif // !GDK_CURSOR_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
