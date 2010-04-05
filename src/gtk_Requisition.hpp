#ifndef GTK_REQUISITION_HPP
#define GTK_REQUISITION_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Requisition
 */
class Requisition
    :
    public Falcon::CoreObject
{
public:

    Requisition( const Falcon::CoreClass*, const GtkRequisition* = 0 );

    ~Requisition();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_REQUISITION_HPP
