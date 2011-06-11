#ifndef GDK_DRAGCONTEXT_HPP
#define GDK_DRAGCONTEXT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::DragContext
 */
class DragContext
    :
    public Gtk::VoidObject
{
public:

    DragContext( const Falcon::CoreClass*, const GdkDragContext* = 0 );

    DragContext( const DragContext& );

    ~DragContext();

    DragContext* clone() const { return new DragContext( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkDragContext* getObject() const { return (GdkDragContext*) m_obj; }

    void setObject( const void* );

private:

    void incref() const;

    void decref() const;

};


} // Gdk
} // Falcon

#endif // !GDK_DRAGCONTEXT_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
