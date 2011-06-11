#ifndef G_PARAMSPEC_HPP
#define G_PARAMSPEC_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Glib {

/**
 *  \class Falcon::Glib::ParamSpec
 */
class ParamSpec
    :
    public Gtk::VoidObject
{
public:

    ParamSpec( const Falcon::CoreClass*, const GParamSpec* = 0 );

    ParamSpec( const ParamSpec& );

    ~ParamSpec();

    ParamSpec* clone() const { return new ParamSpec( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GParamSpec* getObject() const { return (GParamSpec*) m_obj; }

    void setObject( const void* );

private:

    void incref() const;

    void decref() const;

};


} // Glib
} // Falcon

#endif // !G_PARAMSPEC_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
